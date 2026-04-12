from __future__ import annotations


def detect_anomalies(series: list[dict], *, interval_width: float) -> list[dict]:
    if len(series) < 2:
        return []
    anomalies = []
    for previous, current in zip(series, series[1:]):
        residual = current["value"] - previous["value"]
        if abs(residual) > interval_width * 2:
            anomalies.append(
                {
                    "period": current["period"],
                    "actual": current["value"],
                    "forecast": previous["value"],
                    "residual": residual,
                    "severity": "high" if abs(residual) > interval_width * 3 else "medium",
                    "direction": "spike" if residual > 0 else "dip",
                }
            )
    return anomalies
