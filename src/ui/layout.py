"""
Streamlit layout and UI components.
"""

import streamlit as st

from src.config.settings import settings


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] > div:first-child {
            overflow: hidden !important;  /* Đảm bảo không cuộn trong nội dung */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    """Render the sidebar with controls and information."""
    with st.sidebar:
        st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
        st.markdown("---")
        st.subheader("Controls")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            if "chat_manager" in st.session_state:
                st.session_state.chat_manager.clear_conversation()
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        Đây là chatbot AI thông minh giúp bạn gợi ý món ăn phù hợp theo sở thích và ngữ cảnh.

        **Chức năng:**
        - Nhận diện giọng nói
        - Nhận diện ngữ cảnh
        - Gợi ý món ăn
        - Lưu lịch sử ăn uống
        """)

        # st.markdown("---")
        # st.caption(f"Powered by {settings.APP_ICON}")


def render_header():
    """Render the main header."""
    st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
    st.markdown("Ask me anything! I'm here to help.")
    st.markdown("---")
