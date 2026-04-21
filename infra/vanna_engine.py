from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from infra.nim_gateway import nim_gateway
from infra.settings import settings

try:  # pragma: no cover - optional dependency in lightweight test environments
    from vanna.base import VannaBase
    from vanna.chromadb import ChromaDB_VectorStore

    VANNA_AVAILABLE = True
except Exception:  # pragma: no cover - keep module importable even if Vanna is unavailable
    VannaBase = object
    ChromaDB_VectorStore = object
    VANNA_AVAILABLE = False


SEMANTIC_MISS_PATTERNS = (
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "not enough information",
    "no relevant",
    "no data",
    "unable to generate",
    "define a new metric",
)


class VannaSemanticError(RuntimeError):
    """Raised when Vanna cannot ground a question to a safe SQL query."""


@dataclass(slots=True)
class SemanticCacheEntry:
    question: str
    sql: str
    selected_tables: list[str]
    created_at: str


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _extract_sql(payload: Any) -> str:
    text = str(payload or "").strip()
    if not text:
        return ""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip().rstrip(";")


def _looks_like_sql(text: str) -> bool:
    return bool(re.match(r"^(select|with)\b", text.strip(), flags=re.IGNORECASE))


def _is_semantic_miss(text: str) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in SEMANTIC_MISS_PATTERNS):
        return True
    return not _looks_like_sql(text)


if VANNA_AVAILABLE:

    class NIMVanna(ChromaDB_VectorStore, VannaBase):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            ChromaDB_VectorStore.__init__(self, config=config or {})
            VannaBase.__init__(self, config=config or {})

        def system_message(self, message: str) -> dict[str, str]:
            return {"role": "system", "content": message}

        def user_message(self, message: str) -> dict[str, str]:
            return {"role": "user", "content": message}

        def assistant_message(self, message: str) -> dict[str, str]:
            return {"role": "assistant", "content": message}

        def submit_prompt(self, prompt, **kwargs) -> str:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import httpx

                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(
                            f"{settings.nim_base_url.rstrip('/')}/chat/completions",
                            headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
                            json={
                                "model": settings.nim_model,
                                "messages": messages,
                                "temperature": 0.01,
                                "max_tokens": 800,
                            },
                        )
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]
                return asyncio.run(nim_gateway.chat(messages, temperature=0.01))
            except Exception as exc:  # pragma: no cover - depends on external NIM
                raise RuntimeError(f"Vanna NIM submission failed: {exc}") from exc

else:

    class NIMVanna:  # pragma: no cover - only used when dependency is missing
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self.config = config or {}

        def train(self, *args, **kwargs) -> None:
            return None

        def generate_sql(self, question: str) -> str:
            raise RuntimeError("The `vanna` package is not installed.")


def init_vanna() -> NIMVanna:
    path = settings.data_dir / "chromadb"
    path.mkdir(parents=True, exist_ok=True)
    return NIMVanna(config={"path": str(path)})


vn = init_vanna()


class VannaSemanticLayer:
    def __init__(self, instance: NIMVanna = vn) -> None:
        self.vn = instance

    def train_on_ddl(self, ddl: str) -> None:
        if hasattr(self.vn, "train"):
            self.vn.train(ddl=ddl)

    async def generate_sql(self, question: str) -> str:
        raw = self.vn.generate_sql(question=question)
        sql = _extract_sql(raw)
        if not sql or _is_semantic_miss(sql):
            raise VannaSemanticError("Vanna could not ground the request to a known metric.")
        return sql


vanna_engine = VannaSemanticLayer()


class VannaSemanticCache:
    def __init__(self, path: Path | None = None, *, train_successes: bool = True) -> None:
        self.path = path or (settings.data_dir / "cache" / "vanna_semantic_cache.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.train_successes = train_successes

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"entries": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {"entries": {}}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def lookup(self, question: str) -> SemanticCacheEntry | None:
        key = _normalize_question(question)
        payload = self._read()
        entry = payload.get("entries", {}).get(key)
        if not isinstance(entry, dict):
            return None
        return SemanticCacheEntry(
            question=str(entry.get("question", question)),
            sql=str(entry.get("sql", "")),
            selected_tables=[str(item) for item in entry.get("selected_tables", [])],
            created_at=str(entry.get("created_at", _utc_now())),
        )

    def remember(self, question: str, sql: str, selected_tables: list[str]) -> None:
        key = _normalize_question(question)
        payload = self._read()
        entries = payload.setdefault("entries", {})
        entries[key] = asdict(
            SemanticCacheEntry(
                question=question,
                sql=sql,
                selected_tables=[str(item) for item in selected_tables],
                created_at=_utc_now(),
            )
        )
        self._write(payload)
        if self.train_successes and hasattr(vn, "train"):
            try:
                vn.train(question=question, sql=sql)
            except Exception:
                pass

    def invalidate(self, question: str | None = None, sql: str | None = None) -> None:
        payload = self._read()
        entries = payload.get("entries", {})
        if not isinstance(entries, dict) or not entries:
            return

        target_key = _normalize_question(question) if question else None
        keys_to_remove: list[str] = []
        for key, entry in entries.items():
            if target_key is not None and key == target_key:
                keys_to_remove.append(key)
                continue
            if sql and isinstance(entry, dict) and str(entry.get("sql", "")).strip() == str(sql).strip():
                keys_to_remove.append(key)
        for key in keys_to_remove:
            entries.pop(key, None)
        payload["entries"] = entries
        self._write(payload)
