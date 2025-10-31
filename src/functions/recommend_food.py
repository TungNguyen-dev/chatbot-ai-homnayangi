import streamlit as st
from openai import OpenAI

from src.config.settings import settings
from src.context.embeddings import EmbeddingsManager
from src.core.prompt_builder import PromptBuilder
from src.utils.detect_ingredients import detect_ingredients
from src.utils.detect_user_type import detect_user_type
from src.utils.get_meal_time import get_meal_time_from_hour

DEFINITION = {
    "type": "function",
    "function": {
        "name": "recommend_food",
        "description": (
            "Gợi ý món ăn phù hợp dựa trên bối cảnh hội thoại của người dùng, "
            "bao gồm loại người dùng (cá nhân hoặc gia đình), nguyên liệu sẵn có, "
            "và thời điểm trong ngày. "
            "Hàm này được gọi khi người dùng hỏi về việc nên ăn gì, "
            "ví dụ: 'Hôm nay ăn gì', 'Tối nay ăn gì', hoặc khi hội thoại có câu hỏi tương tự."
        ),
        "examples": [
            {
                "user_input": "Tối nay cả nhà ăn gì được?",
                "detected_context": {
                    "serving_type": "family",
                    "ingredients": "thịt bò, rau cải",
                    "meal_type": "evening"
                },
                "example_call": {
                    "name": "recommend_food",
                    "arguments": {
                        "serving_type": "family",
                        "ingredients": "thịt bò, rau cải",
                        "meal_type": "evening"
                    },
                },
                "example_output": {
                    "recommendation": "Lẩu bò hoặc bò xào rau cải là lựa chọn ngon miệng và phù hợp cho bữa tối gia đình."
                },
            },
        ],
    },
}


def handle(llm_client, args: dict):
    """
    Handle user's food recommendation request.

    This function builds a contextual prompt to recommend suitable foods
    based on user's messages, serving type, available ingredients, and meal time.
    If the context is insufficient (e.g., unknown serving type), the function asks
    for clarification instead of calling the model.
    """
    print(st.session_state.messages)
    try:
        # === 1. Detect serving type ===
        serving_type = detect_user_type(llm_client, {"message": st.session_state.messages})
        if serving_type == "unknown":
            return "Bạn muốn chuẩn bị bữa ăn cho cá nhân hay cho gia đình?"

        # === 2. Build dynamic prompt ===
        prompt_parts = ["Recommend foods based on user's preferences",
                        f"- Serving type: {serving_type}"]

        # Detect ingredients (if mentioned by the user)
        ingredients = detect_ingredients(st.session_state.messages)
        if ingredients:
            prompt_parts.append(f"- Ingredients: {ingredients}")

        # Determine the meal type from the current time
        meal_type = get_meal_time_from_hour()
        if meal_type:
            prompt_parts.append(f"- Meal type: {meal_type}")

        # Predefined prompt
        prompt_builder = PromptBuilder()
        sys_prompt = prompt_builder.build_system_message()
        prompt_parts.append(f"- Predefined prompt: {sys_prompt}")

        # History chat
        prompt_parts.append(f"- History chat: {st.session_state.messages}")

        # Combine all parts into a final system prompt
        messageList = []
        system_prompt = "\n".join(prompt_parts)
        messageList.append({"role": "system", "content": system_prompt})

        # Embeddings & RAG
        embeddings = EmbeddingsManager()
        user_message = st.session_state.messages[-1]['content']
        print("User message: " + user_message)
        embeddings.add_text(user_message, metadata={"role": "user"})

        # 🆕 3️⃣ Truy vấn vector DB xem có món nào phù hợp với câu hỏi hoặc sở thích không
        similar_items = []
        if embeddings.enabled:
            similar_items = embeddings.search_similar(user_message, n_results=3)

        # 🆕 4️⃣ Nếu có kết quả, tạo đoạn context để AI dùng
        context_info = ""
        if similar_items:
            context_info = (
                    "Thông tin tham khảo được truy xuất từ cơ sở dữ liệu (có thể hữu ích cho câu hỏi):\n\n"
                    + "\n".join(f"- {item}" for item in similar_items)
            )

        # 5️⃣ Chèn system message chứa context (ưu tiên ngay sau system đầu tiên)
        if context_info:
            # Tạo prompt rõ ràng cho LLM biết cách dùng context
            rag_prompt = {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI chuyên tư vấn về ẩm thực. "
                    "Hãy sử dụng thông tin dưới đây để giúp trả lời câu hỏi người dùng nếu phù hợp.\n\n"
                    f"{context_info}"
                ),
            }
            messageList.append(rag_prompt)

        # === 3. Initialize LLM client ===
        client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        )
        print(messageList)

        # === 4. Stream model response ===
        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messageList,
            temperature=settings.OPENAI_TEMPERATURE,
            max_completion_tokens=settings.OPENAI_MAX_TOKENS,
            stream=True,
        )

        return st.write_stream(stream)

    except Exception as e:
        # Handle unexpected runtime issues gracefully
        print(f"[ERROR] Food recommendation failed: {e}")
        return "Xin lỗi, hệ thống gặp lỗi khi gợi ý món ăn. Vui lòng thử lại sau."
