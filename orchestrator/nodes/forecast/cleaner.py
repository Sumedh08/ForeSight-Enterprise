from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any
from infra.metrics import warning

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.warnings: list[dict[str, str]] = []

    def clean(self) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        """Apply a battery of tests and refinements to the time series."""
        if self.df.empty:
            return self.df, self.warnings

        # 1. Deduplication
        if self.df.index.duplicated().any():
            count = int(self.df.index.duplicated().sum())
            self.warnings.append(warning("data_quality", f"Found {count} duplicate timestamps. Aggregated using mean()."))
            self.df = self.df.groupby(level=0).mean()

        # 2. Outlier Detection & Clipping (Z-score)
        self._clip_outliers()

        # 3. Constant Value Check (Dead Series)
        if self.df["value"].nunique() <= 1 and len(self.df) > 5:
            self.warnings.append(warning("data_quality", "Target series is constant (no variance). Forecasting may be unreliable."))

        # 4. Normalization (Handle Infinity/NaN)
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        return self.df, self.warnings

    def _clip_outliers(self, threshold: float = 4.0):
        """Clip extreme outliers to prevent model distortion."""
        vals = self.df["value"].values
        if len(vals) < 5:
            return

        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0:
            return

        z_scores = np.abs((vals - mean) / std)
        outliers = z_scores > threshold
        
        if outliers.any():
            count = int(outliers.sum())
            self.warnings.append(warning("data_quality", f"Clipped {count} extreme outliers (z-score > {threshold})."))
            
            # Clip values to the threshold boundaries
            upper = mean + threshold * std
            lower = mean - threshold * std
            self.df["value"] = self.df["value"].clip(lower=lower, upper=upper)

def clean_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cleaner = DataCleaner(df)
    return cleaner.clean()
