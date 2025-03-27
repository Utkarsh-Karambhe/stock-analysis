from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from fetch_data import fetch_stock_data, transform_data, load_data_to_db

def etl_pipeline():
    api_key = "91LA1ECVG2H1A36U"
    symbol = "TSLA"
    data = fetch_stock_data(api_key, symbol)
    df = transform_data(data, symbol)
    load_data_to_db(df)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for stock data',
    schedule_interval=timedelta(minutes=1),
)

run_etl = PythonOperator(
    task_id='run_etl',
    python_callable=etl_pipeline,
    dag=dag,
)

run_etl