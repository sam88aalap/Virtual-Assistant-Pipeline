import requests


class WeatherAPIError(Exception):
    """Custom exception for Weather API errors."""
    pass


class WeatherAPI:
    def __init__(self, base_url="https://api.responsible-nlp.net/weather.php", timeout=10):
        self.base_url = base_url
        self.timeout = timeout

    def get_forecast(self, place: str) -> dict:
        """
        Fetch raw weather forecast data as a dictionary.
        """
        try:
            response = requests.post(
                self.base_url,
                data={"place": place},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise WeatherAPIError(f"Request failed: {e}")

        except ValueError:
            raise WeatherAPIError("Invalid JSON response from API")

    def get_forecast_day(self, place: str, day: str) -> dict:
        """
        Fetch raw weather forecast data as a dictionary.
        """
        data = self.get_forecast(place)
        target_day = day.lower()

        for entry in data.get("forecast", []):
            if entry.get("day", "").lower() == target_day:
                return {
                    "place": data.get("place", place),
                    "day": entry.get("day"),
                    "temperature": entry.get("temperature", {}),
                    "weather": entry.get("weather"),
                }

    def get_forecast_text(self, place: str, day: str) -> str:
        """
        Fetch weather forecast and return it as formatted text.
        """
        data = self.get_forecast_day(place, day)

        place = data.get("place")
        day = data.get("day")
        weather = data.get("weather")
        temp_min = data.get("temperature",{}).get("min")
        temp_max = data.get("temperature",{}).get("max")

        lines = [f"Weather forecast for {data.get('place', place)}:"]
        lines.append("-" * 40)

        return f"Weather forecast for {place} on {day}: {weather}, lowest temp of {temp_min} and highest of {temp_max}."


if __name__ == "__main__":
    api = WeatherAPI()
    print(api.get_forecast_text("Marburg", "Monday"))
