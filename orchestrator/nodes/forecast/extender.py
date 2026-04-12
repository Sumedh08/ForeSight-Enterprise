import duckdb
import pandas as pd
from typing import Any
from datetime import datetime
import logging

from infra.settings import settings
from orchestrator.nodes.forecast.model import run_forecast_model

logger = logging.getLogger(__name__)

def attempt_extend_forecast(table_name: str, requested_date: datetime | str) -> bool:
    """
    Check if the requested_date falls outside the current forecast horizon in `{table_name}_forecast`.
    If it does, extend the forecast in the database up to the requested_date.
    Returns True if an extension occurred.
    """
    forecast_table = f"{table_name}_forecast"
    conn = duckdb.connect(str(settings.duckdb_path), read_only=False)
    
    try:
        # Get table schema to find the datetime and numeric columns
        schema = conn.execute(f"DESCRIBE {forecast_table}").df()
        date_col = next((row["column_name"] for _, row in schema.iterrows() if "DATE" in str(row["column_type"]).upper() or "TIMESTAMP" in str(row["column_type"]).upper()), None)
        value_col = next((row["column_name"] for _, row in schema.iterrows() if any(t in str(row["column_type"]).upper() for t in ["DOUBLE", "FLOAT", "DECIMAL", "INTEGER", "BIGINT"]) and row["column_name"] != "is_forecast"), None)
        group_cols = [row["column_name"] for _, row in schema.iterrows() if "VARCHAR" in str(row["column_type"]).upper()]
        
        if not date_col or not value_col:
            return False
            
        # Get max date currently in DB
        max_date_res = conn.execute(f"SELECT MAX({date_col}) FROM {forecast_table}").fetchone()
        if not max_date_res or not max_date_res[0]:
            return False
            
        max_dt = pd.to_datetime(max_date_res[0])
        req_dt = pd.to_datetime(requested_date)
        
        if req_dt <= max_dt:
            # Already have data to satisfy query
            return False
            
        logger.info(f"Requested date {req_dt} exceeds current max {max_dt}. Extending forecasts!")
        
        # Get training data (actual history)
        group_sql = f", {', '.join(group_cols)}" if group_cols else ""
        sql = f"SELECT {date_col} as period, {value_col} as value{group_sql} FROM {forecast_table} WHERE is_forecast = False"
        df = conn.execute(sql).df()
        
        # Calculate roughly how many periods to add to get to req_dt
        freq_guesses = pd.infer_freq(pd.to_datetime(df["period"]).drop_duplicates().sort_values()) or "D"
        diff_days = (req_dt - max_dt).days
        
        # Extra periods needed beyond the existing forecast max_dt
        if "D" in freq_guesses:
            periods_to_add = diff_days
        elif "W" in freq_guesses:
            periods_to_add = (diff_days // 7) + 1
        elif "M" in freq_guesses:
            periods_to_add = (diff_days // 30) + 1
        else:
            periods_to_add = diff_days
            
        # Add a 10% buffer
        periods_to_add = int(periods_to_add * 1.1) + 1
        
        if periods_to_add > 365 * 3:  # Don't extend infinitely
            periods_to_add = 365 * 3
            
        # We need to forecast FROM the end of the history UP TO max_dt + periods_to_add
        # But Bolt requires us to generate a single continuous window from the end of history.
        # Wait, the history ends much earlier than max_dt. So we just need to re-forecast for (existing_horizon + periods_to_add)
        total_history_len = len(df)
        if group_cols:
            groups = df.groupby(group_cols)
            history_len = len(df) // len(groups)
        else:
            history_len = len(df)
            
        # What is the duration from end of history to Max DT?
        # Let's just do a large horizon from history end.
        history_max_dt = df['period'].max()
        total_diff_days = (req_dt - pd.to_datetime(history_max_dt)).days
        if "D" in freq_guesses:
            total_horizon = total_diff_days
        elif "W" in freq_guesses:
            total_horizon = (total_diff_days // 7) + 1
        else:
            total_horizon = (total_diff_days // 30) + 1
            
        total_horizon = int(total_horizon * 1.1) + 1
            
        results = []
        grouped = df.groupby(group_cols) if group_cols else [(None, df)]
        for g_keys, group_df in grouped:
            from orchestrator.nodes.forecast.curator import curate_series
            series = curate_series(group_df.to_dict("records"), grain="day")["series"]
            
            logger.info(f"Extending {g_keys} forecast to horizon {total_horizon}")
            forecast = run_forecast_model(series, horizon=total_horizon, grain="day")
            
            # Store only the NEW points that are > max_dt to append
            for pt in forecast["point_forecast"]:
                pt_date = pd.to_datetime(pt["period"])
                if pt_date > max_dt:
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
                    
        if results:
            result_df = pd.DataFrame(results)
            # Append new points, binding by name to avoid column order issues
            conn.execute(f"INSERT INTO {forecast_table} BY NAME SELECT * FROM result_df")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Failed to extend forecast: {e}")
        return False
    finally:
        conn.close()
