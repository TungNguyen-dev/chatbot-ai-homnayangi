import os
from dotenv import load_dotenv
import speech_recognition as sr

# Load biến môi trường
load_dotenv()

class STTManager:
    @staticmethod
    def transcribe_from_mic(duration: int = 0) -> str:
        """
        Ghi âm từ mic và nhận dạng giọng nói.
        :param duration: thời lượng nói tối đa (giây). 0 = không giới hạn
        """
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            print("🎤 Bắt đầu ghi âm... (nói để bắt đầu, im lặng để dừng)")
            recognizer.adjust_for_ambient_noise(source)

            if duration == 0:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            else:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=duration)

        try:
            print("🧠 Đang xử lý giọng nói...")
            text = recognizer.recognize_google(audio, language="vi-VN")
            text = text.strip().replace("\n", " ").replace("\r", " ")
            print(f"🗣️ Bạn nói: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Không nghe rõ, vui lòng thử lại.")
            return ""
        except sr.RequestError as e:
            print(f"⚠️ Lỗi khi gọi Google API: {e}")
            return ""
