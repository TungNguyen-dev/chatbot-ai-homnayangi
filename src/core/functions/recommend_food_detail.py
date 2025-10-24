DEFINITION = {
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
}


def handle(dispatcher, args: dict) -> str:
    return dispatcher.recommend_food_detail(args.get("style"), args.get("taste"))
