from transformers import pipeline
import numpy as np
import soundfile as sf
 
def text_to_speech(text, output_path="output.wav", model_name="facebook/mms-tts-vie"):
    try:
        # Khởi tạo TTS model
        tts = pipeline("text-to-speech", model=model_name)
        output = tts(text)
 
        audio = output.get("audio", None)
        sampling_rate = output.get("sampling_rate", 22050)
        if audio is None:
            print("Không lấy được audio từ output:", output)
            return None
 
        print("Kiểu dữ liệu audio:", type(audio))
        print("Dạng mảng:", np.shape(audio))
 
        # Nếu là numpy array
        if isinstance(audio, np.ndarray):
            # Đảm bảo mảng là 1 chiều
            if audio.ndim > 1:
                audio = audio.squeeze()
            # Nếu giá trị không nằm trong [-1, 1], chuẩn hóa
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            audio = audio.astype(np.float32)
            sf.write(output_path, audio, sampling_rate, format='WAV', subtype='PCM_16')
            print(f"✅ Đã ghi file WAV thành công: {output_path}")
            return output_path
 
        # Nếu audio là bytes
        elif isinstance(audio, bytes):
            with open(output_path, "wb") as f:
                f.write(audio)
            print(f"✅ Đã ghi file (bytes): {output_path}")
            return output_path
 
        # Nếu audio là list
        elif isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            sf.write(output_path, audio, sampling_rate, format='WAV', subtype='PCM_16')
            print(f"✅ Đã ghi file (list): {output_path}")
            return output_path
 
        else:
            print("❌ Kiểu dữ liệu audio không hỗ trợ:", type(audio))
            return None
 
    except Exception as e:
        print("❌ Lỗi TTS:", e)
        return None
 