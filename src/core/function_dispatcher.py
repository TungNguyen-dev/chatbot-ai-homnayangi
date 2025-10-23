"""
Centralized function calling and dispatching logic for LLM tools.
"""

import json
from typing import Dict, Generator, Optional, Any
from src.core.calling_external_api import Get_weather


# ============================================================
# Tool Function Definitions
# ============================================================

FUNCTION_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "how_to_cook_food",
            "description": "Explain how to cook a specific food",
            "parameters": {
                "type": "object",
                "properties": {
                    "food_name": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["food_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_food",
            "description": "Recommend food based on gender, location, disease, or time",
            "parameters": {
                "type": "object",
                "properties": {
                    "gender": {"type": "string"},
                    "location": {"type": "string"},
                    "disease": {"type": "string"},
                    "time": {"type": "string"},
                },
            },
        },
    },
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_food_recommendation",
            "description": "Recommend food based on location and weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "weather_condition": {"type": "string"},
                },
                "required": ["location", "weather_condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_food_detail",
            "description": "Recommend detailed food style and taste.",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {"type": "string"},
                    "taste": {"type": "string"},
                },
            },
        },
    },
]


# ============================================================
# Dispatcher Class
# ============================================================

class FunctionDispatcher:
    """
    Handles parsing of tool calls, business logic, and LLM prompt utilities.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

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

        # After stream ends, call the corresponding function
        if function_name:
            yield from self.dispatch(function_name, arguments_buffer)

    # --------------------------------------------------------
    # Dispatcher
    # --------------------------------------------------------

    def dispatch(self, function_name: Optional[str], arguments_buffer: str) -> Generator[str, None, None]:
        """Dispatch correct local function after stream ends."""
        if not function_name or not arguments_buffer:
            return

        try:
            args = json.loads(arguments_buffer)
        except json.JSONDecodeError:
            yield "Lỗi khi phân tích tham số hàm từ mô hình."
            return

        handler_map = {
            "how_to_cook_food": lambda: self.how_to_cook(args.get("food_name"), args.get("location")),
            "recommend_food_detail": lambda: self.recommend_food_detail(args.get("style"), args.get("taste")),
            "recommend_food": lambda: self.recommend_food(
                args.get("disease"), args.get("location"), args.get("time"), args.get("gender")
            ),
            "get_food_recommendation": lambda: self.recommend_food_with_weather(
                args.get("weather_condition"), args.get("location")
            ),
            "find_restaurants": lambda: self.find_restaurants(args.get("location"), args.get("cuisine")),
            "get_current_weather": self._handle_weather,
        }

        handler = handler_map.get(function_name)
        if handler:
            yield handler()
        else:
            yield f"⚠️ Function '{function_name}' chưa được hỗ trợ."

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
        # TODO: integrate with ChromaDB logic or knowledge base
        prompt = f"Gợi ý món ăn {style} với hương vị {taste} (Vietnamese)."
        return self._simple_llm_call(prompt)

    def find_restaurants(self, location: Optional[str], cuisine: Optional[str] = None) -> str:
        """Find restaurants, auto-fill location if missing."""
        if not location or location.lower() == "none":
            weather_info = Get_weather().get_location_and_weather()
            if weather_info:
                location = weather_info.get("city")
        prompt = f"Gợi ý nhà hàng {cuisine or ''} tại {location} (Vietnamese)."
        return self._simple_llm_call(prompt)

    # --------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------

    def _simple_llm_call(self, prompt: str) -> str:
        """
        Internal helper for running a simple one-shot LLM query.
        Kept inside dispatcher since it’s only used here.
        """
        response = self.llm_client._chat_completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _handle_weather(self) -> str:
        """Fetch weather information."""
        try:
            weather_info: Optional[Dict[str, Any]] = Get_weather().get_location_and_weather()
            if not weather_info:
                return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."

            city = weather_info.get("city")
            temperature = weather_info.get("temperature")

            if city is None or temperature is None:
                return "Thông tin thời tiết không đầy đủ, vui lòng thử lại sau."

            return f"Thời tiết ở {city} hôm nay là {temperature} độ."
        except Exception:
            return "Hiện tại không thể lấy thông tin thời tiết, vui lòng thử lại sau."

