import requests


class WeatherAPIError(Exception):
    """Custom exception for Weather API errors."""
    pass


class WeatherAPI:
    def __init__(self, base_url="https://api.responsible-nlp.net/weather.php", timeout=10):
        self.base_url = base_url
        self.timeout = timeout

    def get_forecast(self, place: str) -> dict:
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

        return {"error": "Day not found"}

    def get_forecast_text(self, place: str) -> str:
        data = self.get_forecast(place)

        lines = [f"Weather forecast for {data.get('place', place)}:"]
        lines.append("-" * 40)

        for day in data.get("forecast", []):
            day_name = day.get("day", "Unknown day").capitalize()
            temp = day.get("temperature", {})
            min_temp = temp.get("min", "?")
            max_temp = temp.get("max", "?")
            weather = day.get("weather", "Unknown")

            lines.append(
                f"{day_name}: {weather}, "
                f"{min_temp}°C – {max_temp}°C"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    api = WeatherAPI()
    print(api.get_forecast_text("Marburg"))
    print(api.get_forecast_day("Marburg", "Monday"))
