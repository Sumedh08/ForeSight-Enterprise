from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from typing import Any
from infra.metrics import warning
from orchestrator.nodes.forecast.cleaner import clean_time_series

logger = logging.getLogger(__name__)

def curate_series(raw: list[dict], grain: str) -> dict:
    """
    Advanced series curation:
    1. Automatic frequency detection.
    2. Anomaly cleaning (Z-score).
    3. Missing value interpolation.
    4. Quality guardrails.
    """
    df = pd.DataFrame(raw)
    if df.empty:
        return {
            "series": [],
            "warnings": [warning("data_quality", "No historical data was found for the selected metric.")],
            "missing_rate": 1.0,
        }

    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period").sort_index()
    
    # 1. Clean & Refine (Aggregates duplicates, clips outliers)
    df, cleaner_warnings = clean_time_series(df)
    
    warnings_list = cleaner_warnings
    
    # 2. Intelligent Frequency Detection
    # If the user-provided grain is used, we use it, but we can also infer.
    freq_map = {
        "day": "D", 
        "week": "W-MON", 
        "month": "ME", 
        "quarter": "QE",
        "year": "YE"
    }
    target_freq = freq_map.get(grain)
    
    if not target_freq and len(df) > 3:
        # Infer frequency if not provided
        inferred = pd.infer_freq(df.index)
        if inferred:
            target_freq = inferred
            logger.info(f"Inferred frequency: {inferred}")
        else:
            target_freq = "D" # Default
    
    target_freq = target_freq or "D"

    # 3. Regularize Grid
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=target_freq)
    
    # Only reindex if the grid actually differs or has gaps
    df = df.reindex(full_index)
    
    # 4. Handle Gaps
    missing_rate = float(df["value"].isna().mean()) if len(df.index) else 1.0
    if df["value"].isna().any():
        missing_count = int(df["value"].isna().sum())
        if missing_rate > 0.4:
            warnings_list.append(warning("data_quality", f"High missing data rate ({missing_rate*100:.1f}%). Forecast accuracy will be low."))
        
        warnings_list.append(warning("data_quality", f"{missing_count} missing periods were auto-filled."))
        df["value"] = df["value"].interpolate(method="linear").ffill().bfill()

    series_data = [
        {"period": idx.to_pydatetime(), "value": float(value)} 
        for idx, value in df["value"].items()
    ]

    return {
        "series": series_data,
        "warnings": warnings_list,
        "missing_rate": missing_rate,
        "final_grain": grain
    }
