import os
import requests
from dotenv import load_dotenv
from datetime import datetime, UTC
import csv


def fetch_weather(api_key: str, city="Moscow") -> dict:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    weather_data = response.json()

    # Извлекаем нужные данные
    dt_unix = weather_data.get("dt")
    dt_str = datetime.fromtimestamp(dt_unix, UTC).strftime('%Y-%m-%d %H:%M:%S') if dt_unix else ""
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



if __name__ == "__main__":
    # test weather API
    load_dotenv(".env")
    api_key = os.getenv("API_KEY")
    city = "Moscow"
    data = fetch_weather(api_key=api_key, city=city)
    print(data)
