DEFINITION = {
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
}


def handle(dispatcher, args: dict) -> str:
    return dispatcher.recommend_food(
        args.get("disease"), args.get("location"), args.get("time"), args.get("gender")
    )
