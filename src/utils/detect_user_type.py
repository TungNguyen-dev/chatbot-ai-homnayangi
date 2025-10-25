DEFINITION = {
    "type": "function",
    "function": {
        "name": "detect_user_type",
        "description": "Determine whether the food request is intended for an individual or a family context.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The user's original message asking for food recommendations.",
                },
            },
            "required": ["message"],
        },
    },
}


def detect_user_type(llm_client, args: dict) -> str:
    message = args["message"]

    prompt = (
        "Analyze the following message and respond with only one word: "
        "'personal' if the request is for an individual, or 'family' if it's for a group or household.\n\n"
        f"Message: {message}\n\n"
        "Answer:"
    )

    response = llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}])
    answer = response.choices[0].message.content.strip().lower()

    # Normalize output to ensure consistent return
    if "family" in answer:
        return "family"
    return "personal"
