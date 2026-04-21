from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.settings import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()


def tokenize_terms(value: str) -> set[str]:
    text = str(value).strip().lower()
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    return {token for token in re.findall(r"[a-z]+|\d+", text) if token}


def _frame_columns(frame: Any) -> list[str]:
    columns = getattr(frame, "columns", None)
    if columns is not None:
        return [str(column) for column in columns]
    if isinstance(frame, list) and frame and isinstance(frame[0], dict):
        return [str(column) for column in frame[0].keys()]
    return []


def _frame_values(frame: Any, column_name: str) -> list[Any]:
    if hasattr(frame, "__getitem__") and hasattr(frame, "columns"):
        series = frame[column_name]
        if hasattr(series, "tolist"):
            return list(series.tolist())
        return list(series)
    if isinstance(frame, list):
        return [row.get(column_name) for row in frame if isinstance(row, dict)]
    return []


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for pattern in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _datetime_ratio(values: list[Any]) -> float:
    usable = [value for value in values if value not in (None, "")]
    if not usable:
        return 0.0
    parsed = sum(1 for value in usable if _parse_datetime(value) is not None)
    return float(parsed) / float(len(usable))


def _numeric_ratio(values: list[Any]) -> float:
    usable = [value for value in values if value not in (None, "")]
    if not usable:
        return 0.0
    parsed = 0
    for value in usable:
        try:
            float(value)
            parsed += 1
        except Exception:
            continue
    return float(parsed) / float(len(usable))


def detect_temporal_column(frame: Any) -> str | None:
    best: tuple[float, str] | None = None
    for column_name in _frame_columns(frame):
        values = _frame_values(frame, column_name)
        lowered = str(column_name).lower()
        score = _datetime_ratio(values)
        if any(token in lowered for token in ("date", "time", "timestamp", "period", "month", "week", "year", "day", "observation")):
            score += 0.35
        if best is None or score > best[0]:
            best = (score, str(column_name))
    if best is None or best[0] < 0.5:
        return None
    return best[1]


def detect_metric_columns(frame: Any, *, date_column: str | None = None) -> list[str]:
    metrics: list[tuple[float, str]] = []
    for column_name in _frame_columns(frame):
        if date_column and str(column_name) == date_column:
            continue
        lowered = str(column_name).lower()
        if lowered in {"id", "index"} or lowered.endswith("_id"):
            continue
        score = _numeric_ratio(_frame_values(frame, column_name))
        if any(token in lowered for token in ("value", "rate", "amount", "sales", "revenue", "price", "count", "total", "balance", "metric")):
            score += 0.25
        if score >= 0.75:
            metrics.append((score, str(column_name)))
    metrics.sort(key=lambda item: (-item[0], item[1]))
    return [column_name for _, column_name in metrics]


@dataclass(slots=True)
class TrainingJobRecord:
    ingestion_id: str
    table_name: str
    date_column: str
    value_column: str
    series_id: str
    predictor_name: str
    warehouse_profile_id: str
    warehouse_profile_name: str
    state: str
    progress_pct: int
    message: str
    poll_after_ms: int
    created_at: str
    updated_at: str
    last_error: str | None = None

    def as_summary(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "predictor_name": self.predictor_name,
            "state": self.state,
            "progress_pct": self.progress_pct,
            "message": self.message,
            "poll_after_ms": self.poll_after_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingJobRecord":
        return cls(
            ingestion_id=str(payload.get("ingestion_id", "")),
            table_name=str(payload.get("table_name", "")),
            date_column=str(payload.get("date_column", "")),
            value_column=str(payload.get("value_column", "")),
            series_id=str(payload.get("series_id", "")),
            predictor_name=str(payload.get("predictor_name", "")),
            warehouse_profile_id=str(payload.get("warehouse_profile_id", "")),
            warehouse_profile_name=str(payload.get("warehouse_profile_name", "")),
            state=str(payload.get("state", "queued")),
            progress_pct=int(payload.get("progress_pct", 0)),
            message=str(payload.get("message", "")),
            poll_after_ms=int(payload.get("poll_after_ms", 3000)),
            created_at=str(payload.get("created_at", utc_now_iso())),
            updated_at=str(payload.get("updated_at", utc_now_iso())),
            last_error=payload.get("last_error"),
        )


def build_training_jobs(
    *,
    frame: Any,
    table_name: str,
    ingestion_id: str | None,
    warehouse_profile: dict[str, Any],
    poll_after_ms: int = 3000,
) -> list[TrainingJobRecord]:
    resolved_ingestion_id = ingestion_id or str(uuid.uuid4())
    date_column = detect_temporal_column(frame)
    if not date_column:
        return []

    metric_columns = detect_metric_columns(frame, date_column=date_column)
    if not metric_columns:
        return []

    created_at = utc_now_iso()
    jobs: list[TrainingJobRecord] = []
    for metric_column in metric_columns:
        series_id = f"{table_name}.{metric_column}"
        predictor_name = sanitize_identifier(f"{table_name}_{metric_column}_predictor")
        jobs.append(
            TrainingJobRecord(
                ingestion_id=resolved_ingestion_id,
                table_name=table_name,
                date_column=date_column,
                value_column=metric_column,
                series_id=series_id,
                predictor_name=predictor_name,
                warehouse_profile_id=str(warehouse_profile.get("id", "")),
                warehouse_profile_name=str(warehouse_profile.get("name", "enterprise-warehouse")),
                state="queued",
                progress_pct=5,
                message=f"Queued Lightwood training for `{metric_column}`.",
                poll_after_ms=poll_after_ms,
                created_at=created_at,
                updated_at=created_at,
            )
        )
    return jobs


class TrainingStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (settings.data_dir / "cache" / "training_jobs.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"jobs": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"jobs": []}
        except Exception:
            return {"jobs": []}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_jobs(self) -> list[TrainingJobRecord]:
        payload = self._read()
        jobs = [TrainingJobRecord.from_dict(item) for item in payload.get("jobs", []) if isinstance(item, dict)]
        jobs.sort(key=lambda item: item.updated_at, reverse=True)
        return jobs

    def save_jobs(self, jobs: list[TrainingJobRecord]) -> None:
        payload = self._read()
        indexed = {
            item.get("series_id"): TrainingJobRecord.from_dict(item)
            for item in payload.get("jobs", [])
            if isinstance(item, dict) and item.get("series_id")
        }
        for job in jobs:
            indexed[job.series_id] = job
        self._write({"jobs": [asdict(record) for record in indexed.values()]})

    def update_job(self, series_id: str, **changes: Any) -> TrainingJobRecord | None:
        jobs = self.list_jobs()
        target: TrainingJobRecord | None = None
        for job in jobs:
            if job.series_id != series_id:
                continue
            target = job
            break
        if target is None:
            return None
        for key, value in changes.items():
            if hasattr(target, key):
                setattr(target, key, value)
        target.updated_at = utc_now_iso()
        self.save_jobs(jobs)
        return target

    def get_job(self, *, series_id: str) -> TrainingJobRecord | None:
        for job in self.list_jobs():
            if job.series_id == series_id:
                return job
        return None

    def get_jobs_by_ingestion(self, ingestion_id: str) -> list[TrainingJobRecord]:
        return [job for job in self.list_jobs() if job.ingestion_id == ingestion_id]

    def match_job(
        self,
        *,
        question: str,
        metric_hint: str | None = None,
        table_names: list[str] | None = None,
    ) -> TrainingJobRecord | None:
        question_tokens = tokenize_terms(question)
        hint_tokens = tokenize_terms(metric_hint or "")
        best: tuple[int, TrainingJobRecord] | None = None
        allowed_tables = {table.lower() for table in (table_names or [])}

        for job in self.list_jobs():
            if allowed_tables and job.table_name.lower() not in allowed_tables:
                continue
            job_tokens = (
                tokenize_terms(job.table_name)
                | tokenize_terms(job.value_column)
                | tokenize_terms(job.series_id)
                | tokenize_terms(job.predictor_name)
            )
            score = len(question_tokens & job_tokens) * 2
            score += len(hint_tokens & job_tokens) * 3
            if any(token in job.value_column.lower() for token in ("value", "rate", "amount", "price", "sales")):
                score += 1
            if best is None or score > best[0]:
                best = (score, job)
        if best is None or best[0] <= 0:
            return None
        return best[1]
