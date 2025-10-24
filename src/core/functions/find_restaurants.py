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


def handle(dispatcher, args: dict) -> str:
    return dispatcher.find_restaurants(args.get("location"), args.get("cuisine"))
