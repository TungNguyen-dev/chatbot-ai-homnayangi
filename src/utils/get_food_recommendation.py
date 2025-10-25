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
    location = args.get("location")
    weather_condition = args.get("weather_condition")
    prompt = (
        f"Gợi ý món ăn ngon ở {location} dựa trên cảm giác khi trời {weather_condition} độ C."
    )
    response = dispatcher.llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
