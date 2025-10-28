import json
import logging
import re

import langdetect  # pip install langdetect

logger = logging.getLogger(__name__)
DEFINITION = {
    "type": "function",
    "function": {
        "name": "recommend_food_detail",
        "description": (
            "Gợi ý chi tiết món ăn dựa trên phong cách ẩm thực (style), "
            "hương vị (taste) và dịp ăn uống (occasion). "
            "Hàm này được dùng khi người dùng hỏi 'hôm nay ăn gì', "
            "hoặc yêu cầu gợi ý món ăn cụ thể."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "description": "Phong cách ẩm thực, ví dụ: Việt Nam, Nhật Bản, Hàn Quốc, Âu..."
                },
                "taste": {
                    "type": "string",
                    "description": "Hương vị mong muốn, ví dụ: cay, ngọt, chua, mặn, béo, thanh đạm..."
                },
                "count": {
                    "type": "integer",
                    "description": "Số lượng món ăn muốn gợi ý (mặc định: 1).",
                    "default": 1
                },
                "occasion": {
                    "type": "string",
                    "description": "Dịp ăn uống, ví dụ: bữa sáng, bữa trưa, tiệc, ngày lạnh, ngày nóng, lễ tết...",
                    "default": "thường ngày"
                }
            },
            "required": ["style", "taste"]
        },
    },
}


def detect_language(text: str) -> str:
    """Tự động phát hiện ngôn ngữ người dùng."""
    try:
        lang = langdetect.detect(text)
        return "vi" if lang == "vi" else "en"
    except Exception:
        return "vi"  # fallback mặc định là tiếng Việt


def handle(llm_client, args: dict, user_input: str = "") -> str:
    """
    Hàm được gọi khi GPT muốn gợi ý món ăn chi tiết.
    Hỗ trợ đa ngôn ngữ, nhiều món, dịp ăn uống.
    """
    style = args.get("style")
    taste = args.get("taste")
    count = args.get("count", 1)
    occasion = args.get("occasion", "thường ngày")
    if not style or not taste:
        return "⚠️ Bạn cần cung cấp đủ thông tin về phong cách (style) và hương vị (taste)."
    user_lang = detect_language(user_input or f"{style} {taste}")
    prompt = f"""
   Bạn là một chuyên gia ẩm thực quốc tế.
   Hãy gợi ý {count} món ăn theo phong cách **{style}**, có hương vị **{taste}**, phù hợp cho dịp **{occasion}**.
   Trả kết quả ở định dạng JSON:
   [
     {{
       "ten_mon": "Tên món ăn",
       "thanh_phan_chinh": "Nguyên liệu chính",
       "cach_che_bien": "Mô tả ngắn gọn cách chế biến (1-2 câu)",
       "ly_do_phu_hop": "Lý do vì sao món ăn này phù hợp với hương vị {taste}"
     }}
   ]
   Mỗi món <= 50 từ. Ngôn ngữ: {"Tiếng Việt" if user_lang == "vi" else "Tiếng Anh"}.
   """
    try:
        response = llm_client._chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
        raw_content = response.choices[0].message.content
        cleaned = re.sub(r"^```(json)?", "", raw_content.strip())
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("⚠️ GPT trả JSON không hợp lệ. Dùng raw text.")
            data = None
        if data and isinstance(data, list):
            formatted = "\n\n".join([
                f"🍽️ **{item.get('ten_mon', 'Món ăn')}**\n"
                f"• Thành phần chính: {item.get('thanh_phan_chinh', '')}\n"
                f"• Cách chế biến: {item.get('cach_che_bien', '')}\n"
                f"• Lý do phù hợp: {item.get('ly_do_phu_hop', '')}"
                for item in data
            ])
        else:
            formatted = raw_content
        # Nếu người dùng nói tiếng Anh → dịch kết quả sang tiếng Anh
        if user_lang == "en":
            translation_prompt = f"Translate this text into fluent English:\n{formatted}"
            translated = llm_client.chat_completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": translation_prompt}],
                temperature=0.5,
                max_tokens=600,
            )
            formatted = translated.choices[0].message.get("content", "").strip()
        return formatted or "😔 Không tìm được món ăn phù hợp, hãy thử lại nhé."
    except Exception as e:
        logger.error(f"[ERROR] recommend_food_detail: {e}")
        return "❌ Có lỗi xảy ra khi gợi ý món ăn. Vui lòng thử lại sau."
