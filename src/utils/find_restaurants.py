import requests

DEFINITION = {
    "type": "function",
    "function": {
        "name": "find_restaurants",
        "description": "Find restaurants based on cuisine and location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "cuisine": {"type": "string"},
            },
        },
    },
}


def _get_location_and_weather():
    try:
        location_url = "http://ipwhois.app/json/"
        location_response = requests.get(location_url, timeout=5)
        location_response.raise_for_status()
        location_data = location_response.json()
        city = location_data.get("city", "Unknown")
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_response = requests.get(weather_url, timeout=5)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        temperature = weather_data["current_weather"]["temperature"]
        return {"city": city, "temperature": temperature, "latitude": lat, "longitude": lon}
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching location/weather: {e}")
        return None


def handle(dispatcher, args: dict) -> str:
    location = args.get("location")
    cuisine = args.get("cuisine")
    if not location or str(location).lower() == "none":
        weather_info = _get_location_and_weather()
        if weather_info:
            location = weather_info.get("city")
    prompt = f"Gợi ý nhà hàng {cuisine or ''} tại {location} (Vietnamese)."
    response = dispatcher.llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
