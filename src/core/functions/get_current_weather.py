DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the user's current city and temperature using free APIs.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def handle(dispatcher, args: dict | None = None) -> str:
    # No args required
    return dispatcher._handle_weather()
