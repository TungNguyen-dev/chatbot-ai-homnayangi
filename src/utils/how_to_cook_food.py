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
    food_name = args.get("food_name")
    location = args.get("location")
    prompt = (
        f"Briefly explain how {food_name} is prepared in {location} (Vietnamese, ≤5 lines)."
        if location
        else f"Briefly explain how {food_name} is prepared (Vietnamese, ≤5 lines)."
    )
    response = dispatcher.llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
