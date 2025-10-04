"""
Main Streamlit application entry point for the chatbot.
"""

import streamlit as st
from src.ui import setup_page_config
from src.ui.chat_ui import render_chat_interface
from src.core.chat_manager import ChatManager
from src.utils import setup_logger

# Setup logger
logger = setup_logger(__name__)


def main():
    """Main application entry point."""
    # Setup page configuration
    setup_page_config()

    # Initialize chat manager
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    # Render chat interface
    render_chat_interface(st.session_state.chat_manager)


if __name__ == "__main__":
    main()
