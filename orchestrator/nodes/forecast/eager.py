import logging
import pandas as pd
import duckdb
from typing import Any
import asyncio

from infra.settings import settings
from infra.semantic_layer import semantic_layer
from orchestrator.nodes.forecast.curator import curate_series
from orchestrator.nodes.forecast.model import run_forecast_model

logger = logging.getLogger(__name__)

async def generate_eager_forecasts(table_name: str, date_col: str, value_col: str, group_cols: list[str]):
    """
    Background job to eagerly forecast a dataset up to 3 years and save it back to DuckDB.
    """
    logger.info(f"Starting eager forecasting for {table_name}.{value_col} grouping by {group_cols}")
    
    # 1. Connect to DuckDB and retrieve data
    conn = duckdb.connect(str(settings.duckdb_path), read_only=False)
    
    try:
        # Get historical data
        group_sql = f", {', '.join(group_cols)}" if group_cols else ""
        sql = f"SELECT {date_col} AS period, {value_col} AS value{group_sql} FROM {table_name} WHERE {value_col} IS NOT NULL"
        df = conn.execute(sql).df()
        
        # Determine frequency to calculate max 3 year horizon
        if len(df) < 5:
            return # Too small to auto-forecast
            
        freq_guesses = pd.infer_freq(pd.to_datetime(df["period"]).drop_duplicates().sort_values())
        freq_guesses = freq_guesses or "D"
        
        # 3 years max horizon
        multiplier = 3
        if "D" in freq_guesses:
            max_horizon = 365 * multiplier
        elif "W" in freq_guesses:
            max_horizon = 52 * multiplier
        elif "M" in freq_guesses:
            max_horizon = 12 * multiplier
        elif "Q" in freq_guesses:
            max_horizon = 4 * multiplier
        elif "A" in freq_guesses or "Y" in freq_guesses:
            max_horizon = 3 # 3 years
        else:
            max_horizon = 100
        
        # 2. Split by groups if necessary and forecast
        forecast_table_name = f"{table_name}_forecast"
        
        # Instead of just overriding, we append both History and Forecast to the forecast table
        # so Vanna can query seamlessly.
        
        results = []
        if group_cols:
            grouped = df.groupby(group_cols)
        else:
            grouped = [(None, df)]
            
        for g_keys, group_df in grouped:
            history_len = len(group_df)
            horizon = min(history_len, max_horizon)  # Forecast length = history length, max 3 years
            
            raw_series = group_df.to_dict("records")
            
            # Curate
            prepared = curate_series(raw_series, grain="day") # It internally auto-detects
            series = prepared["series"]
            
            # Forecast (Bolt)
            logger.info(f"Forecasting {table_name}.{value_col} group {g_keys} for {horizon} periods.")
            forecast = run_forecast_model(series, horizon=horizon, grain="day")
            
            # Store point forecasts
            for pt in forecast["point_forecast"]:
                row = {
                    date_col: pt["period"],
                    value_col: pt["value"],
                    "is_forecast": True
                }
                if group_cols:
                    if isinstance(g_keys, tuple):
                        for c, k in zip(group_cols, g_keys):
                            row[c] = k
                    else:
                        row[group_cols[0]] = g_keys
                results.append(row)
                
            # Store history natively marked as not forecast
            for pt in series:
                row = {
                    date_col: pt["period"],
                    value_col: pt["value"],
                    "is_forecast": False
                }
                if group_cols:
                    if isinstance(g_keys, tuple):
                        for c, k in zip(group_cols, g_keys):
                            row[c] = k
                    else:
                        row[group_cols[0]] = g_keys
                results.append(row)
                
        # 3. Save to DuckDB
        result_df = pd.DataFrame(results)
        conn.execute(f"DROP TABLE IF EXISTS {forecast_table_name}")
        conn.execute(f"CREATE TABLE {forecast_table_name} AS SELECT * FROM result_df")
        
        logger.info(f"Eager forecasting complete. Created {forecast_table_name} with {len(result_df)} rows.")
        
    except Exception as e:
        logger.error(f"Eager forecast failed: {e}")
    finally:
        conn.close()
