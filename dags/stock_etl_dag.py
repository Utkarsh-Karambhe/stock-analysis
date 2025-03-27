import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Set absolute path to scripts directory
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, scripts_path)

try:
    from fetch_data import fetch_stock_data, transform_data, load_data_to_db
except ImportError as e:
    raise ImportError(f"Failed to import modules: {e}. Check sys.path: {sys.path}")

def etl_pipeline():
    api_key = "MJJTJOTUY0HJKO68"  # Consider using Airflow Variables for this
    symbol = "UBER"
    
    try:
        data = fetch_stock_data(api_key, symbol)
        if not data or "Time Series (1min)" not in data:
            raise ValueError("No valid data received from API")
        
        df = transform_data(data, symbol)
        load_data_to_db(df)
    except Exception as e:
        print(f"ETL Pipeline failed: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 23),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

dag = DAG(
    'stock_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for stock data',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['stocks'],
)

run_etl = PythonOperator(
    task_id='run_etl',
    python_callable=etl_pipeline,
    dag=dag,
)