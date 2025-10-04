"""
Streamlit layout and UI components.
"""

import streamlit as st
from src.config.settings import APP_TITLE, APP_ICON


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar():
    """Render the sidebar with controls and information."""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
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
        This is an AI-powered chatbot that can help you with various tasks and questions.
        
        **Features:**
        - Contextual conversations
        - Memory of chat history
        - Helpful and accurate responses
        """)

        st.markdown("---")
        st.caption(f"Powered by {APP_ICON}")


def render_header():
    """Render the main header."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Ask me anything! I'm here to help.")
    st.markdown("---")
