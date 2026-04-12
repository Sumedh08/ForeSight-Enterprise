"""
Dynamic MindsDB Predictor Orchestrator
Connects MindsDB to Postgres and auto-trains predictors for discovered time-series tables.
"""
import time
import requests
import json
from sqlalchemy import create_engine, text

# Configurations
MINDSDB_API = "http://localhost:47334/api"
PG_HOST = "postgres" # Internal docker network
PG_DB = "natwest_db"
PG_USER = "admin"
PG_PASS = "adminpassword"

# Local PG connection for discovery
LOCAL_PG = "postgresql://admin:adminpassword@localhost:5432/natwest_db"

def setup_mindsdb_datasource():
    """Tells MindsDB how to connect to our Postgres database."""
    print("Connecting MindsDB to Postgres...")
    payload = {
        "name": "natwest_datasource",
        "engine": "postgres",
        "connection_data": {
            "user": PG_USER,
            "password": PG_PASS,
            "host": PG_HOST,
            "port": 5432,
            "database": PG_DB
        }
    }
    # Using SQL via API for more reliability in setup
    sql = f"""
    CREATE DATABASE IF NOT EXISTS natwest_datasource
    WITH ENGINE = 'postgres',
    PARAMETERS = {{
        "user": "{PG_USER}",
        "password": "{PG_PASS}",
        "host": "{PG_HOST}",
        "port": 5432,
        "database": "{PG_DB}"
    }};
    """
    resp = requests.post(f"{MINDSDB_API}/sql", json={"query": sql})
    print("Datasource setup status:", resp.status_code, resp.text)

def discover_and_train():
    """Finds tables in Postgres and creates Predictors in MindsDB automatically."""
    engine = create_engine(LOCAL_PG)
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
        
    for (table_name,) in tables:
        if table_name.endswith("_forecast") or table_name == "alembic_version":
            continue
            
        print(f"Analyzing table for forecasting: {table_name}")
        # Build predictor name
        predictor_name = f"{table_name}_predictor"
        
        # Check if predictor already exists
        check_sql = f"SELECT * FROM mindsdb.predictors WHERE name = '{predictor_name}'"
        check_resp = requests.post(f"{MINDSDB_API}/sql", json={"query": check_sql}).json()
        
        if len(check_resp.get("data", [])) > 0:
            print(f"Predictor {predictor_name} already exists. Skipping.")
            continue

        # Dynamic Column discovery
        with engine.connect() as conn:
            cols = conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")).fetchall()
            
        date_col = None
        target_col = None
        
        for c_name, c_type in cols:
            c_name_lower = c_name.lower()
            # Find date-like column
            if any(k in c_type.lower() for k in ("timestamp", "date")) or c_name_lower in ("date", "ds", "timestamp", "period"):
                date_col = c_name
            # Find first numeric column as target
            if not target_col and any(k in c_type.lower() for k in ("numeric", "int", "double", "precision", "real")):
                if c_name_lower not in ("id", "index"):
                    target_col = c_name

        if date_col and target_col:
            print(f"Found candidate for forecast: {table_name} (Time: {date_col}, Target: {target_col})")
            
            # Create the predictor in MindsDB
            train_sql = f"""
            CREATE PREDICTOR mindsdb.{predictor_name}
            FROM natwest_datasource
              (SELECT * FROM {table_name})
            PREDICT {target_col}
            ORDER BY {date_col}
            WINDOW 20
            HORIZON 30;
            """
            print(f"Triggering training for {predictor_name}...")
            train_resp = requests.post(f"{MINDSDB_API}/sql", json={"query": train_sql})
            print("Training trigger status:", train_resp.status_code)
        else:
            print(f"Table {table_name} does not meet time-series requirements (Date: {date_col}, Target: {target_col})")

if __name__ == "__main__":
    # In a real setup, we'd wait for MindsDB to be 'ready'
    # For now, we assume this is called by Airflow or manual trigger after infra is up.
    try:
        setup_mindsdb_datasource()
        discover_and_train()
    except Exception as e:
        print("Error in dynamic MindsDB setup:", e)
