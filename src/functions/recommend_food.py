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
                "user_type": {"type": "string", "enum": ["personal", "family"]},
            },
        },
    },
}


def handle(llm_client, args: dict) -> str:
    disease = args.get("disease")
    location = args.get("location")
    time = args.get("time")
    gender = args.get("gender")
    user_type = args.get("user_type")
    prompt = (
        f"Recommend dishes for a {user_type} user with {disease}, {location}, {time}, {gender} (Vietnamese)."
    )
    response = llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
