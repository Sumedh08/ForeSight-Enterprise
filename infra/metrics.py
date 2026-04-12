from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WarningPayload:
    kind: str
    message: str


def warning(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": message}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def sql_confidence(
    *,
    retrieval_score: float,
    selector_margin: float,
    validation_ok: bool,
    dry_run_ok: bool,
    execution_ok: bool,
    repair_attempts: int,
) -> float:
    score = 0.2
    score += 0.2 * clamp(retrieval_score)
    score += 0.2 * clamp(selector_margin)
    score += 0.15 if validation_ok else -0.15
    score += 0.15 if dry_run_ok else -0.2
    score += 0.15 if execution_ok else -0.3
    score -= 0.07 * repair_attempts
    return round(clamp(score), 2)


def forecast_confidence(
    *,
    history_points: int,
    missing_rate: float,
    beats_baseline: bool,
    coverage_80: float | None,
) -> float:
    """Confidence score for zero-shot foundation models (Chronos/TimesFM)."""
    score = 0.40  # Start higher for Foundation Models
    score += min(history_points, 12) / 12 * 0.30  # Bonus for 12+ points
    score -= clamp(missing_rate) * 0.2
    score += 0.15 if beats_baseline else -0.1
    if coverage_80 is not None:
        # Penalize less for low coverage on very small backtest sets
        target = 0.8
        score += (coverage_80 - target) * 0.3
    return round(clamp(score), 2)
