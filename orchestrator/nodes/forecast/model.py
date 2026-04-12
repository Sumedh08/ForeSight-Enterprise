from __future__ import annotations

from infra.metrics import warning
from orchestrator.nodes.forecast.baseline import seasonal_naive_forecast


def run_forecast_model(series: list[dict], *, horizon: int, grain: str) -> dict:
    """Zero-shot forecasting using Amazon Chronos-Bolt-Small via BaseChronosPipeline."""
    try:
        import torch
        import pandas as pd
        from chronos import BaseChronosPipeline

        # Prepare 1D tensor of values
        values = torch.tensor([item["value"] for item in series])
        
        # BaseChronosPipeline auto-detects Chronos vs ChronosBolt architecture
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu", 
            dtype=torch.float32,
        )

        # ChronosBolt supports quantile levels within [0.1 - 0.9] training range
        quantile_levels = [0.1, 0.2, 0.5, 0.8, 0.9]
        result = pipeline.predict_quantiles(
            values.unsqueeze(0),
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )
        # predict_quantiles returns (quantiles_tensor, mean_tensor)
        # quantiles_tensor shape: (batch, horizon, num_quantiles)
        quantile_tensor = result[0] if isinstance(result, tuple) else result

        # Build future date index
        last_date = series[-1]["period"]
        freq_map = {"W": "W", "D": "D", "M": "ME", "Y": "YE",
                    "week": "W", "day": "D", "month": "ME", "year": "YE", "quarter": "QE"}
        future_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=freq_map.get(grain, "D")
        )[1:]

        # Extract quantiles: indices map to quantile_levels [0.1, 0.2, 0.5, 0.8, 0.9]
        q = quantile_tensor.squeeze(0)  # shape: (horizon, 5)
        low_80 = q[:, 0]    # 0.1 quantile (80% lower)
        low_95 = q[:, 0]    # 0.1 is our best approximation for 95% lower 
        median  = q[:, 2]   # 0.5 quantile (median)
        high_80 = q[:, 4]   # 0.9 quantile (80% upper)
        high_95 = q[:, 4]   # 0.9 is our best approximation for 95% upper

        point_forecast = [
            {"period": date.to_pydatetime(), "value": float(val)}
            for date, val in zip(future_dates, median)
        ]
        
        prediction_intervals = [
            {
                "period": date.to_pydatetime(),
                "low_80": float(l80),
                "high_80": float(h80),
                "low_95": float(l95),
                "high_95": float(h95),
            }
            for date, l80, h80, l95, h95 in zip(future_dates, low_80, high_80, low_95, high_95)
        ]

        return {
            "point_forecast": point_forecast,
            "prediction_intervals": prediction_intervals,
            "warnings": [],
        }

    except ImportError as e:
        raise RuntimeError(f"Missing dependency for zero-shot forecasting: {e}. Run 'pip install chronos-forecasting torch'")
    except Exception as e:
        raise RuntimeError(f"Zero-shot inference failed: {e}")
