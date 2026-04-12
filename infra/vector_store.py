from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from infra.settings import settings


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(text_tokens))


@dataclass(slots=True)
class RetrievalResult:
    id: str
    score: float
    payload: dict


class RetrievalStore:
    def __init__(self, examples_path: Path | None = None, schema_cache_path: Path | None = None) -> None:
        self.examples_path = examples_path or settings.examples_path
        self.schema_cache_path = schema_cache_path or settings.schema_cache_path
        self.examples = json.loads(self.examples_path.read_text(encoding="utf-8")) if self.examples_path.exists() else []
        self.schema_cache = (
            json.loads(self.schema_cache_path.read_text(encoding="utf-8"))
            if self.schema_cache_path.exists()
            else {"details": {}}
        )

    def health(self) -> str:
        return "up" if self.examples else "degraded"

    def retrieve_examples(self, question: str, top_k: int = 3) -> list[dict]:
        ranked = sorted(
            (
                RetrievalResult(id=item["id"], score=lexical_score(question, item["question"]), payload=item)
                for item in self.examples
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return [item.payload for item in ranked[:top_k] if item.score > 0]

    def retrieve_cell_values(self, question: str, top_k: int = 6) -> list[dict]:
        values: list[dict] = []
        for table_name, table_detail in self.schema_cache.get("details", {}).items():
            for column_name, sample_values in (table_detail.get("sample_values") or {}).items():
                for value in sample_values:
                    text = f"{table_name} {column_name} {value}"
                    score = lexical_score(question, text)
                    if score > 0:
                        values.append(
                            {
                                "table": table_name,
                                "column": column_name,
                                "value": value,
                                "score": score,
                            }
                        )
        values.sort(key=lambda item: item["score"], reverse=True)
        return values[:top_k]


retrieval_store = RetrievalStore()
