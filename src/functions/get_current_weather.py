import requests

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the user's current city and temperature using free APIs.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _get_location_and_weather():
    """Get a user's city and weather using public APIs."""
    try:
        # Step 1: Get location via IPWhois API
        location_url = "http://ipwhois.app/json/"
        location_response = requests.get(location_url, timeout=5)
        location_response.raise_for_status()
        location_data = location_response.json()

        city = location_data.get("city", "Unknown")
        district = location_data.get("region", "Unknown")
        country = location_data.get("country_name", "Unknown")
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")

        # Step 2: Get weather from Open-Meteo
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_response = requests.get(weather_url, timeout=5)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        temperature = weather_data["current_weather"]["temperature"]

        return {
            "district": district,
            "city": city,
            "country": country,
            "temperature": temperature,
            "latitude": lat,
            "longitude": lon,
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching location/weather: {e}")
        return None


def handle(dispatcher, args: dict | None = None) -> str:
    # No args required
    try:
        weather_info = _get_location_and_weather()
        if not weather_info:
            return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."
        city = weather_info.get("city")
        temperature = weather_info.get("temperature")
        if city is None or temperature is None:
            return "Thông tin thời tiết không đầy đủ, vui lòng thử lại sau."
        return f"Thời tiết ở {city} hôm nay là {temperature}°C."
    except Exception:
        return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."
