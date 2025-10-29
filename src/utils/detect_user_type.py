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
        "Based on the entire conversation below, determine the main intent of the messages. "
        "Classify whether the conversation is primarily intended for:\n"
        "- an individual → respond 'personal'\n"
        "- a family or group → respond 'family'\n"
        "- unclear or insufficient context → respond 'unknown'\n\n"
        "Respond with exactly one word: 'personal', 'family', or 'unknown'. "
        "Do not include any explanation or punctuation.\n\n"
        f"Conversation:\n{messages}\n\n"
        "Answer:"
    )

    response = llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip().lower()
