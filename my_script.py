def my_function():
    print("Hello, Airflow!")


from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from my_script import my_function

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your.email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_dag',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=timedelta(days=1),
    start_date=datetime.utcnow(),
    tags=['example'],
)

run_my_function = PythonOperator(
    task_id='print_hello',
    python_callable=my_function,
    dag=dag,
)