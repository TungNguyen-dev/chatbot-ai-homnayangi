import streamlit as st
from openai import OpenAI

from src.config.settings import settings
from src.context.embeddings import EmbeddingsManager
from src.core.prompt_builder import PromptBuilder
from src.utils.detect_ingredients import detect_ingredients
from src.utils.detect_user_type import detect_user_type
from src.utils.get_meal_time import get_meal_time_from_hour

# === 🧩 Function Definition Metadata ===
DEFINITION = {
    "type": "function",
    "function": {
        "name": "recommend_food",
        "description": (
            "Gợi ý món ăn phù hợp dựa trên bối cảnh hội thoại của người dùng, "
            "bao gồm loại người dùng (cá nhân hoặc gia đình), nguyên liệu sẵn có, "
            "và thời điểm trong ngày. "
            "Hàm này được gọi khi người dùng hỏi về việc nên ăn gì, "
            "ví dụ: 'Hôm nay ăn gì', 'Tối nay ăn gì', hoặc các câu hỏi tương tự."
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
                    "recommendation": (
                        "**Gợi ý món ăn:** Lẩu bò hoặc bò xào rau cải\n"
                        "**Lý do đề xuất:** Phù hợp cho bữa tối gia đình, dễ nấu và đầy đủ dinh dưỡng.\n"
                        "**Cách chuẩn bị:** Dùng thịt bò thái mỏng, nấu với rau cải hoặc xào cùng hành tỏi."
                    )
                },
            },
        ],
    },
}


# === ⚙️ Core Function ===
def handle(llm_client, args: dict):
    """
    Generate a food recommendation based on conversation context.

    Context includes:
      - User type (personal/family)
      - Mentioned ingredients
      - Time of day (meal type)
      - Conversation history
      - Vector search context (if available)
    """
    try:
        messages = []
        history = st.session_state.messages
        user_message = history[-1]["content"]

        # === 1️⃣ Base System Instruction ===
        prompt_builder = PromptBuilder()
        predefined_prompt = prompt_builder.build_system_message()

        system_intro = (
            "You are a culinary assistant AI specializing in Vietnamese cuisine. "
            "Your task is to recommend dishes that are delicious, practical, "
            "and suitable for the user's situation.\n\n"
            "Always consider:\n"
            "- Who will eat (personal or family)\n"
            "- Mentioned or available ingredients\n"
            "- The current meal time (breakfast, lunch, dinner)\n\n"
            "If information is missing, politely ask for clarification "
            "instead of making assumptions.\n\n"
            "---\n"
            "### 📋 RESPONSE FORMAT (output in Vietnamese)\n"
            "Always respond using the following structure:\n\n"
            "**Gợi ý món ăn:** <tên món chính hoặc thực đơn>\n"
            "**Lý do đề xuất:** <vì sao món này phù hợp>\n"
            "**Cách chuẩn bị (nếu cần):** <gợi ý sơ lược về cách chế biến hoặc nguyên liệu bổ sung>\n\n"
            "If there are multiple suitable dishes, list up to 3 options using bullet points."
        )

        messages.append({
            "role": "system",
            "content": f"{system_intro}\n\n---\nPredefined prompt:\n{predefined_prompt}"
        })

        # === 2️⃣ Detect Serving Type ===
        serving_type = detect_user_type(llm_client, {"message": history})
        if serving_type == "unknown":
            return "Bạn muốn chuẩn bị bữa ăn cho cá nhân hay cho gia đình?"
        messages.append({"role": "system", "content": f"- Serving type: {serving_type}"})

        # === 3️⃣ Add Conversation History ===
        messages.append({"role": "system", "content": f"- Conversation history: {history}"})

        # === 4️⃣ Detect Ingredients ===
        ingredients = detect_ingredients(history)
        if ingredients:
            messages.append(
                {"role": "system", "content": f"- Ingredients mentioned: {ingredients}"})

        # === 5️⃣ Determine Meal Type ===
        meal_type = get_meal_time_from_hour()
        if meal_type:
            messages.append({"role": "system", "content": f"- Meal time: {meal_type}"})

        # === 6️⃣ Retrieve Vector Context (RAG) ===
        context_info = ""
        embeddings = EmbeddingsManager()
        if embeddings.enabled:
            # Chỉ lưu nếu có "tôi thích", "tôi muốn", hoặc chứa tên món ăn
            if any(keyword in user_message.lower() for keyword in ["tôi thích", "tôi muốn", "muốn", "thích"]):
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
        print(context_info)

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

            # Chèn vào sau message system đầu tiên
            if messages and messages[0]["role"] == "system":
                messages.insert(1, rag_prompt)
            else:
                messages.insert(0, rag_prompt)

        # === 7️⃣ Final User Message ===
        messages.append({"role": "user", "content": user_message})

        # === 8️⃣ Generate Streamed Response ===
        client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        )

        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=settings.OPENAI_TEMPERATURE,
            max_completion_tokens=settings.OPENAI_MAX_TOKENS,
            stream=True,
        )

        return st.write_stream(stream)

    except Exception as e:
        print(f"[ERROR] Food recommendation failed: {e}")
        return "Xin lỗi, hệ thống gặp lỗi khi gợi ý món ăn. Vui lòng thử lại sau."
