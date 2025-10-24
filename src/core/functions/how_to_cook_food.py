DEFINITION = {
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
}


def handle(dispatcher, args: dict) -> str:
    return dispatcher.how_to_cook(args.get("food_name"), args.get("location"))
