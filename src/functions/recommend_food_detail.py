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
    style = args.get("style")
    taste = args.get("taste")
    prompt = f"Gợi ý món ăn {style} với hương vị {taste} (Vietnamese)."
    response = dispatcher.llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
