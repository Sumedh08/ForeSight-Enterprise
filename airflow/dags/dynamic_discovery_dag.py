from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'natwest',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dynamic_data_discovery',
    default_args=default_args,
    description='Automatically discovers new tables and triggers MindsDB training',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['natwest', 'dynamic'],
) as dag:

    # Task 1: Check for new tables and setup MindsDB Predictors
    # We run the script inside the airflow container or via bash if accessible
    sync_mindsdb = BashOperator(
        task_id='sync_mindsdb_predictors',
        bash_command='python3 /opt/airflow/plugins/mindsdb_dynamic_setup.py',
    )

    # Task 2: Trigger a Cubejs reload if needed 
    # (Note: Cubejs in dev mode reloads automatically, but we can hit the reload API in prod)
    refresh_cube = BashOperator(
        task_id='refresh_semantic_layer',
        bash_command='curl -X GET http://cubejs:4000/cubejs-api/v1/reload || echo "CubeJS reload skipped"',
    )

    sync_mindsdb >> refresh_cube
