DEFINITION = {
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
}


def handle(dispatcher, args: dict) -> str:
    return dispatcher.recommend_food_with_weather(
        args.get("weather_condition"), args.get("location")
    )
