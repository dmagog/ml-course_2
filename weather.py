from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import csv


def fetch_weather(api_key: str, city="Moscow") -> dict:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    weather_data = response.json()

    # Извлекаем нужные данные
    dt_unix = weather_data.get("dt")
    dt_str = datetime.fromtimestamp(dt_unix, timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if dt_unix else ""
    weather_main = weather_data.get("weather", [{}])[0].get("main", "")
    weather_description = weather_data.get("weather", [{}])[0].get("description", "")
    temp = weather_data.get("main", {}).get("temp", "")
    feels_like = weather_data.get("main", {}).get("feels_like", "")
    pressure = weather_data.get("main", {}).get("pressure", "")
    wind_speed = weather_data.get("wind", {}).get("speed", "")

    # Подготовим строку для записи
    row = [dt_str, city, weather_main, weather_description, temp, feels_like, pressure, wind_speed]

    # Проверим, существует ли файл, и если нет — добавим заголовки
    file_exists = os.path.isfile("weather.csv")
    
    with open("weather.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["datetime", "city", "weather_main", "weather_description",
                             "temp", "feels_like", "pressure", "wind_speed"])
        writer.writerow(row)

    return weather_data



# if __name__ == "__main__":
#     # test weather API
#     load_dotenv(".env")
#     api_key = os.getenv("API_KEY")
#     city = "Moscow"
#     data = fetch_weather(api_key=api_key, city=city)
#     print(data)

# Основной код для DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 6),
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    'email_on_failure': False,
    'email_on_retry': False,
    'email_on_success': False,
    'depends_on_past': False,
}

dag = DAG(
    'fetch_weather',
    default_args=default_args,
    description='Fetch weather data every minute',
    schedule_interval='*/1 * * * *',  # каждую минуту
    start_date=datetime(2025, 5, 7),
    catchup=False,
    tags=['weather'],
    
)

# Задачи

fetch_weather_task = PythonOperator(
    task_id='fetch_weather',
    python_callable=fetch_weather,
    op_kwargs={
            "api_key": os.getenv("API_KEY")
        },
    provide_context=True,
    dag=dag,
)

# Задачи выполняются последовательно
fetch_weather_task