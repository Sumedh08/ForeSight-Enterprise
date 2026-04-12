from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from mindsdb_dynamic_setup import run_from_conf


def run_mindsdb_sync(**context):
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    connection_profile = conf.get("connection_profile")
    tables = conf.get("tables")
    if isinstance(tables, str):
        try:
            tables = json.loads(tables)
        except Exception:
            tables = [tables]
    return run_from_conf(connection_profile=connection_profile, tables=tables)


default_args = {
    "owner": "natwest",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="dynamic_data_discovery",
    default_args=default_args,
    description="Offline schema discovery and MindsDB predictor synchronization",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["natwest", "offline", "mindsdb"],
) as dag:
    sync_mindsdb = PythonOperator(
        task_id="sync_mindsdb_predictors",
        python_callable=run_mindsdb_sync,
    )

    refresh_cube = BashOperator(
        task_id="refresh_semantic_layer",
        bash_command='curl -s -X GET http://cubejs:4000/cubejs-api/v1/reload || echo "CubeJS reload skipped"',
    )

    sync_mindsdb >> refresh_cube
