DEFINITION = {
    "type": "function",
    "function": {
        "name": "detect_user_type",
        "description": (
            "Xác định loại người dùng dựa trên ngữ cảnh câu hỏi về ăn uống, "
            "phân biệt giữa cá nhân ('personal') và gia đình ('family'). "
            "Hàm này giúp xác định khi người dùng hỏi nên ăn gì, món gì, "
            "dành cho bản thân hay cho cả nhà.\n\n"
            "Quy tắc nhận diện:\n"
            "- Nếu câu chứa nhân xưng cá nhân số ít (tôi, em, anh, chị, bạn, mình...) → user_type = 'personal'.\n"
            "- Nếu câu chứa nhân xưng tập thể hoặc gia đình (chúng tôi, gia đình tôi, nhà tôi, cả nhà, vợ chồng tôi, con tôi...) → user_type = 'family'.\n"
            "- Nếu không xác định rõ, mặc định user_type = 'unknown'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Tin nhắn gốc của người dùng hỏi về việc ăn uống. "
                        "Ví dụ: 'Tối nay ăn gì', 'Cả nhà tôi nên nấu món gì', 'Hôm nay tôi ăn gì'."
                    ),
                },
            },
            "required": ["message"],
        },
        "examples": [
            {
                "user_input": "Tối nay tôi ăn gì?",
                "expected_output": {"user_type": "personal"},
                "reason": "Câu có nhân xưng 'tôi' → cá nhân."
            },
            {
                "user_input": "Cả nhà tôi ăn gì tối nay?",
                "expected_output": {"user_type": "family"},
                "reason": "Câu có cụm 'cả nhà tôi' → gia đình."
            },
            {
                "user_input": "Gia đình tôi nên nấu món gì cuối tuần này?",
                "expected_output": {"user_type": "family"},
                "reason": "Có cụm 'gia đình tôi' → ngữ cảnh tập thể."
            },
        ],
    },
}


def detect_user_type(llm_client, args: dict) -> str:
    messages = args["message"]

    prompt = (
        "You are a precise intent classifier. "
        "Analyze the full conversation below and decide who the food-related message is mainly intended for. "
        "Classify the intent strictly into one of these categories:\n"
        "- 'personal' → clearly directed toward an individual (uses pronouns like 'I', 'me', 'my', 'myself', etc.).\n"
        "- 'family' → clearly directed toward a family or group (mentions 'we', 'our family', 'my family', 'our house', etc.).\n"
        "- 'unknown' → if the message lacks clear context or you cannot confidently decide between 'personal' and 'family'.\n\n"
        "Your response must contain **only one word**: 'personal', 'family', or 'unknown'. "
        "Do not include explanations, punctuation, or any extra text.\n\n"
        f"Conversation:\n{messages}\n\n"
        "Answer:"
    )

    response = llm_client._chat_completion(
        messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip().lower()
