from transformers import pipeline
import numpy as np
import soundfile as sf


def text_to_speech(text, output_path="storage/output.wav", model_name="facebook/mms-tts-vie"):
    try:
        # Initialize TTS model
        tts = pipeline("text-to-speech", model=model_name)

        # Generate audio (returns dict)
        output = tts(text)

        # ✅ Validate structure
        if not isinstance(output, dict):
            print(f"❌ Unexpected output type: {type(output)}")
            return None

        audio = output.get("audio")
        sampling_rate = output.get("sampling_rate")

        if audio is None or sampling_rate is None:
            print(f"❌ Output missing keys: {list(output.keys())}")
            return None

        print(f"✅ Audio generated successfully! Shape: {np.shape(audio)}, Rate: {sampling_rate}")

        # Ensure audio is NumPy array
        audio = np.array(audio, dtype=np.float32)

        # Ensure 1D
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        # Normalize amplitude if necessary
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio /= peak

        # Save to file
        sf.write(output_path, audio, sampling_rate, format="WAV", subtype="PCM_16")

        print(f"✅ File saved successfully: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Lỗi TTS: {e}")
        return None
