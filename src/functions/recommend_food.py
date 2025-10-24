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
    disease = args.get("disease")
    location = args.get("location")
    time = args.get("time")
    gender = args.get("gender")
    prompt = (
        f"Recommend dishes for {disease}, {location}, {time}, {gender} (Vietnamese)."
    )
    response = dispatcher.llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
