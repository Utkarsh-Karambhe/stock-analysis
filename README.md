# Automated Data Pipeline for Real-Time Analytics

## Overview
This project implements an **Automated Data Pipeline** using **Apache Airflow, Python, and PostgreSQL** to fetch stock market data from APIs every minute, store it in a database, and create real-time analytics dashboards. The pipeline is fully containerized using **Docker** and managed via **Airflow**.

## Tech Stack
- **Python** - Data extraction and processing
- **PostgreSQL** - Database for storing real-time data
- **Apache Airflow** - Orchestrating ETL workflow
- **Docker & Docker Compose** - Containerized environment
- **APIs** - Fetching real-time stock data

---
## Features
- **Automated ETL Pipeline**: Extracts data every minute from financial APIs.
- **Data Storage**: Stores extracted data in PostgreSQL.
- **Workflow Orchestration**: Uses Apache Airflow to schedule and monitor jobs.
- **Containerized Setup**: Runs in a fully Dockerized environment.
- **Monitoring & Logging**: Keeps track of job execution with logs.

---
## Setup Instructions

### 1. Clone the Repository
```sh
 git clone https://github.com/Utkarsh-Karambhe/stock-analysis.git
 cd stock-analysis
```

### 2. Install Docker & Docker Compose
Ensure **Docker** and **Docker Compose** are installed on your system.
- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

### 3. Configure Environment Variables
Create a `.env` file and add your API keys (🔑 Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key))
```sh
API_KEY=your_api_key_here
DB_HOST=postgres
DB_USER=airflow
DB_PASSWORD=airflow
DB_NAME=airflow
```

### 4. Start the Services
Run the following command to spin up Airflow and PostgreSQL containers:
```sh
docker-compose up -d
```
This will start:
- **Airflow Scheduler & Webserver**
- **PostgreSQL Database**
- **Real-time data extraction DAG**

### 5. Access Airflow UI
Open [http://localhost:8080](http://localhost:8080) and login:
- **Username:** admin
- **Password:** admin

### 6. Activate DAG
- Navigate to the **DAGs** section in Airflow UI.
- Enable the `real_time_data_pipeline` DAG.

---
## Folder Structure
```
├── dags/                   # Airflow DAGs
│   ├── stock_etl_dag.py    # Main ETL DAG
│   ├── fetch_data.py       # Data extraction script
│   ├── dashboard.py        # Dashboard script
├── logs/                   # Logs directory
├── Dockerfile              # Docker build file
├── Dockerfile.dashboard    # Optional dashboard setup
├── docker-compose.yml      # Docker Compose setup
├── airflow.cfg             # Airflow configuration
├── airflow.db              # Airflow database
├── requirements.txt        # Python dependencies
├── webserver_config.py     # Webserver configuration
└── README.md               # Documentation
```

---
## Code Overview
### **1. ETL Pipeline (`dags/stock_etl_dag.py`):**
```python
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
```

---
## Troubleshooting
### Check Docker Logs
```sh
docker-compose logs airflow_scheduler
```
### Restart Services
```sh
docker-compose down
```
```sh
docker-compose up -d
```
### Check Database Connection
```sh
docker exec -it airflow_postgres psql -U airflow -d airflow
```
---
## Future Improvements
- **Real-time dashboards using Grafana**.
- **More API integrations for diversified data**.
- **Alert system for anomalies in stock prices**.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributors
- **Utkarsh Karambhe** - Initial Implementation

---
## Contact
For any queries, feel free to reach out:
📧 Email: utkarshkarambhe@gmail.com
💼 LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/utkarsh-karambhe-764bb1248/)
🚀 GitHub: [Your GitHub](https://github.com/Utkarsh-Karambhe)
