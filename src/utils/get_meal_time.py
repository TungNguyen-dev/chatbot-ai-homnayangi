from datetime import datetime

def get_meal_time_from_hour():
    """
    Xác định phần của ngày (bữa ăn) dựa trên giờ hiện tại.
    Trả về str mô tả thời điểm: "sáng", "trưa", "chiều", "tối" hoặc "đêm".
    """
    # Lấy giờ hiện tại dưới dạng số nguyên (0‑23)
    hour = datetime.now().hour
    if 5 <= hour < 11:
        return "sáng"
    elif 11 <= hour < 14:
        return "trưa"
    elif 14 <= hour < 17:
        return "chiều"
    elif 17 <= hour < 22:
        return "tối"
    else:
        return "đêm"
