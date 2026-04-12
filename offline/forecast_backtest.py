from __future__ import annotations

import json
from datetime import datetime

from components.forecasting import ForecastingEngine
from infra.connection_profiles import ConnectionProfileManager


def _find_series(connector) -> list[dict]:
    schema = connector.introspect_schema(sample_limit=1)
    for table in schema.tables:
        datetime_columns = []
        numeric_columns = []
        for column in table.columns:
            dtype = column.data_type.lower()
            if any(token in dtype for token in ("date", "time", "timestamp")):
                datetime_columns.append(column.name)
            if any(token in dtype for token in ("int", "double", "float", "numeric", "decimal", "real")):
                numeric_columns.append(column.name)
        if not datetime_columns or not numeric_columns:
            continue
        period_column = datetime_columns[0]
        value_column = numeric_columns[0]
        query = (
            f'SELECT "{period_column}" AS period, "{value_column}" AS value '
            f'FROM "{table.name}" '
            f'WHERE "{period_column}" IS NOT NULL AND "{value_column}" IS NOT NULL '
            f'ORDER BY "{period_column}"'
        )
        rows = connector.execute(query, preview_limit=5000).rows
        series = []
        for row in rows:
            period = row.get("period")
            value = row.get("value")
            if period is None or value is None:
                continue
            if not isinstance(period, datetime):
                try:
                    period = datetime.fromisoformat(str(period).replace("Z", "+00:00"))
                except Exception:
                    continue
            try:
                value = float(value)
            except Exception:
                continue
            series.append({"period": period, "value": value})
        if len(series) >= 24:
            return series
    return []


def run_metric_backtests() -> dict:
    manager = ConnectionProfileManager()
    profile = manager.get_active_profile()
    connector = manager.build_connector(profile, read_only=True)
    real_series = _find_series(connector)
    if not real_series:
        return {
            "status": "blocked",
            "message": "No forecastable time series found in active database.",
        }

    engine = ForecastingEngine()
    synthetic = engine.synthetic_dataset(periods=96, grain="week", horizon=8)
    result = engine.evaluate(synthetic_dataset=synthetic, real_world_series=real_series)
    return {
        "status": "ok",
        "synthetic": {
            "mae": result.synthetic.mae,
            "mape": result.synthetic.mape,
            "baseline_mae": result.synthetic.baseline_mae,
            "baseline_mape": result.synthetic.baseline_mape,
            "beats_baseline": result.synthetic.beats_baseline,
        },
        "real_world": {
            "mae": result.real_world.mae,
            "mape": result.real_world.mape,
            "baseline_mae": result.real_world.baseline_mae,
            "baseline_mape": result.real_world.baseline_mape,
            "beats_baseline": result.real_world.beats_baseline,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_metric_backtests(), indent=2, default=str))
