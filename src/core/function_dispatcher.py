"""
Centralized function calling and dispatching logic for LLM tools.
"""

# ============================================================
# Auto-discovery of tool function handlers from src.core.functions
# ============================================================
import importlib
import json
import pkgutil
from typing import Dict, Generator, Optional, Any

import requests


# ============================================================
# Dispatcher Class
# ============================================================


class FunctionDispatcher:
    """
    Handles parsing of tool calls, business logic, and LLM prompt utilities.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        # Dynamically discover tool function handlers from src.core.functions
        self.function_handlers = self._load_function_handlers()

    def _load_function_handlers(self) -> Dict[str, Any]:
        """
        Auto-discover function handler modules under src.core.functions.
        Each module is expected to expose:
          - DEFINITION: dict with shape {"function": {"name": str, ...}}
          - handle(dispatcher, args) -> str
        Returns a mapping from function name to its handler callable.
        """
        handlers: Dict[str, Any] = {}
        try:
            package = importlib.import_module("src.core.functions")
        except Exception as e:
            print(f"⚠️ Could not import functions package: {e}")
            return handlers

        if not hasattr(package, "__path__"):
            # Not a package, nothing to scan
            return handlers

        for module_info in pkgutil.iter_modules(package.__path__):
            mod_name = module_info.name
            if mod_name.startswith("_"):
                continue
            full_name = f"src.core.functions.{mod_name}"
            try:
                mod = importlib.import_module(full_name)
            except Exception as e:
                print(f"⚠️ Failed to import module {full_name}: {e}")
                continue

            func_name = None
            try:
                definition = getattr(mod, "DEFINITION", None)
                if isinstance(definition, dict):
                    func = definition.get("function") or {}
                    func_name = func.get("name")
            except Exception:
                func_name = None

            handle = getattr(mod, "handle", None)
            if not callable(handle):
                continue

            if not func_name:
                # Fallback to module name
                func_name = mod_name

            handlers[func_name] = handle

        return handlers

    def reload_function_handlers(self) -> None:
        """Reload function handlers by rescanning the package."""
        self.function_handlers = self._load_function_handlers()

    # --------------------------------------------------------
    # Stream Handler
    # --------------------------------------------------------

    def handle_stream(self, stream) -> Generator[str, None, None]:
        """Parse stream chunks and handle both text and function calls."""
        arguments_buffer, function_name = "", None

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
            elif delta.tool_calls:
                tool_call = delta.tool_calls[0]
                func = tool_call.function
                if func.name:
                    function_name = func.name
                if func.arguments:
                    arguments_buffer += func.arguments

        # After the stream ends, call the corresponding function
        if function_name:
            yield from self.dispatch(function_name, arguments_buffer)

    # --------------------------------------------------------
    # Dispatcher
    # --------------------------------------------------------

    def dispatch(self, function_name: Optional[str], arguments_buffer: str) -> Generator[
        str, None, None]:
        """Dispatch correct local function after stream ends."""
        if not function_name or arguments_buffer is None:
            return

        try:
            args = json.loads(arguments_buffer) if arguments_buffer else {}
        except json.JSONDecodeError:
            yield "❌ Error parsing function arguments from model."
            return

        handler = self.function_handlers.get(function_name)
        if handler:
            # Each handler expects (dispatcher, args) and returns str
            yield handler(self, args)
        else:
            yield f"⚠️ Function '{function_name}' is not supported."

    # --------------------------------------------------------
    # Local Function Implementations
    # --------------------------------------------------------

    def recommend_food(self, disease: str, location: str, time: str, gender: str) -> str:
        prompt = f"Recommend dishes for {disease}, {location}, {time}, {gender} (Vietnamese)."
        return self._simple_llm_call(prompt)

    def how_to_cook(self, food_name: str, location: str) -> str:
        prompt = f"Briefly explain how {food_name} is prepared in {location} (Vietnamese, ≤5 lines)."
        return self._simple_llm_call(prompt)

    def recommend_food_with_weather(self, weather_temp: str, location: str) -> str:
        prompt = f"Gợi ý món ăn ngon ở {location} dựa trên cảm giác khi trời {weather_temp} độ C."
        return self._simple_llm_call(prompt)

    def recommend_food_detail(self, style: str, taste: str) -> str:
        prompt = f"Gợi ý món ăn {style} với hương vị {taste} (Vietnamese)."
        return self._simple_llm_call(prompt)

    def find_restaurants(self, location: Optional[str], cuisine: Optional[str] = None) -> str:
        """Find restaurants, auto-fill location if missing."""
        if not location or location.lower() == "none":
            weather_info = self.get_location_and_weather()
            if weather_info:
                location = weather_info.get("city")
        prompt = f"Gợi ý nhà hàng {cuisine or ''} tại {location} (Vietnamese)."
        return self._simple_llm_call(prompt)

    # --------------------------------------------------------
    # Utility: Weather and Location Retrieval
    # --------------------------------------------------------

    def get_location_and_weather(self) -> Optional[dict]:
        """Get user's city and weather using public APIs."""
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

    # --------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------

    def _simple_llm_call(self, prompt: str) -> str:
        """Simple single-turn LLM call."""
        response = self.llm_client._chat_completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""

    def _handle_weather(self) -> str:
        """Fetch weather information and return human-readable text."""
        try:
            weather_info: Optional[Dict[str, Any]] = self.get_location_and_weather()
            if not weather_info:
                return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."

            city = weather_info.get("city")
            temperature = weather_info.get("temperature")

            if city is None or temperature is None:
                return "Thông tin thời tiết không đầy đủ, vui lòng thử lại sau."

            return f"Thời tiết ở {city} hôm nay là {temperature}°C."
        except Exception:
            return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."
