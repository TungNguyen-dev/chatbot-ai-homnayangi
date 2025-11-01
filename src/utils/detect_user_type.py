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
    messages = args["message"]

    prompt = (
        "You are a conversation classifier. "
        "Carefully analyze the entire conversation below and determine who the messages are primarily intended for. "
        "Classify the intent strictly into one of the following categories:\n"
        "- 'personal' → clearly intended for one individual person.\n"
        "- 'family' → clearly intended for a family or group of people.\n"
        "- 'unknown' → if there is any uncertainty, ambiguity, or insufficient context to confidently choose between 'personal' and 'family'.\n\n"
        "Respond with exactly one word: 'personal', 'family', or 'unknown'. "
        "Do not include any explanation, punctuation, or additional text.\n\n"
        f"Conversation:\n{messages}\n\n"
        "Answer:"
    )

    response = llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip().lower()
