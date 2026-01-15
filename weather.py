#### NOT part of the project ####


import requests
from config import OPENWEATHER_API_KEY

def get_weather(city):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    resp = requests.get(url)
    data = resp.json()

    if resp.status_code != 200:
        return "Couldn't fetch weather data."

    return f"It is {data['main']['temp']}°C with {data['weather'][0]['description']} in {city}."

if __name__ == "__main__":
 print(get_weather('Kochi'))

# test code to inspect api components 
""" city='Marburg'
url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
response = requests.get(url).json()
print(response) """