import requests
class Get_weather:
    def __init__(self):
        pass
    def get_location_and_weather(self):
        """Get location and weather information using free APIs"""
        try:
            # Get location from IP
            print("🌍 Getting your location...")
            # Option 1 (current): ip-api.com
            # location_url = "http://ip-api.com/json/"

            # Option 2: ipwhois.app
            location_url = "http://ipwhois.app/json/"
            location_response = requests.get(location_url, timeout=5)
            location_response.raise_for_status()
            location_data = location_response.json()

            city = location_data.get('city', 'Unknown')
            district = location_data.get('region', 'Unknown')
            country = location_data.get('country_name', 'Unknown')
            lat = location_data.get('latitude')
            lon = location_data.get('longitude')

            # Get weather using Open-Meteo (free, no API key needed)
            print("🌡️  Getting weather information...")
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            weather_response = requests.get(weather_url, timeout=5)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            temperature = weather_data['current_weather']['temperature']

            return {
                'district': district,
                'city': city,
                'country': country,
                'temperature': temperature,
                'latitude': lat,
                'longitude': lon
            }

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
