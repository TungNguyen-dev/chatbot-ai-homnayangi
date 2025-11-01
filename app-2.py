import streamlit as st
from openai import OpenAI
 
from src.config.settings import settings
from src.core.llm_client import LLMClient
from src.utils.stt_manager import STTManager
 
from src.ui.layout import setup_page_config, render_sidebar, render_header
# st.title("Hôm nay ăn gì?")
# --- "Clear Conversation" ---
setup_page_config()
 
# 2. Render Sidebar và Header (giống app.py)
# Sidebar chứa nút "Clear Conversation"
render_sidebar()
# Header sẽ hiển thị tiêu đề chính của ứng dụng
render_header()
 
client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=settings.OPENAI_API_KEY,
)
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 
# def handle_prompt(prompt: str):
#     llm_client = LLMClient()
#     return st.write_stream(llm_client.generate_response_stream(st.session_state.messages))
 
def handle_prompt(prompt: str):
    llm_client = LLMClient()
    response = ""
    for chunk in llm_client.generate_response_stream(st.session_state.messages):
        response += chunk
        # st.markdown(chunk)
       
    return response
 
# --- CSS ghim mic xuống bottom ---
st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        position: fixed !important;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        max-width: 800px;
        background-color: white;
        padding: 0.5rem 1rem 1rem 1rem;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
        border-radius: 12px 12px 0 0;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)
 
 
# --- Ô nhập text và nút mic cùng hàng ---
col1, col2 = st.columns([9, 1])
with col1:
    prompt = st.chat_input("Nhập câu cần hỏi?")
with col2:
    mic_clicked = st.button("🎙️", help="Nói bằng giọng nói", use_container_width=True)
 
 
# Normal
if mic_clicked:
    with st.spinner("🎧 Đang nghe..."):
        spoken_text = STTManager.transcribe_from_mic(duration=0)
    if spoken_text:
        st.session_state.messages.append({"role": "user", "content": spoken_text})
        with st.chat_message("user"):
            st.markdown(spoken_text)
 
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Đang suy nghĩ..._")
            response = handle_prompt(prompt)
            placeholder.markdown(response)
            # response = handle_prompt(spoken_text)
            # st.markdown(response)
 
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
 
 
# --- Khi người dùng nhập text ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Đang suy nghĩ..._")
        response = handle_prompt(prompt)
        placeholder.markdown(response)
        # response = handle_prompt(prompt)
 
        if not response or response.strip() == "":
            response = "⚠️ Không nhận được phản hồi, vui lòng thử lại."
        # st.markdown(response)
 
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()