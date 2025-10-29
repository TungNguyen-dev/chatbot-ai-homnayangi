import streamlit as st
from openai import OpenAI

from src.config.settings import settings
from src.core.llm_client import LLMClient
from src.utils.stt_manager import STTManager

st.title("Hôm nay ăn gì?")

client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=settings.OPENAI_API_KEY,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def handle_prompt(prompt: str):
    llm_client = LLMClient()
    return st.write_stream(llm_client.generate_response_stream(st.session_state.messages))


# Normal
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = handle_prompt(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Speech-to-text input
if st.button("🎙️ Nói bằng giọng nói"):
    with st.spinner("Đang nghe..."):
        spoken_text = STTManager.transcribe_from_mic(duration=0)
    if spoken_text:
        st.session_state.messages.append({"role": "user", "content": spoken_text})
        with st.chat_message("user"):
            st.markdown(spoken_text)

        # Chatbot trả lời
        with st.chat_message("assistant"):
            response = handle_prompt(spoken_text)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
