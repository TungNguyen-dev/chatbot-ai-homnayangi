"""
Chat UI components for rendering messages and handling interactions.
"""

from typing import TYPE_CHECKING

import streamlit as st

from src.utils.tts import text_to_speech

if TYPE_CHECKING:
    from src.core.chat_manager import ChatManager

from src.ui.layout import render_sidebar, render_header


def render_chat_interface(chat_manager: "ChatManager"):
    """
    Render the complete chat interface.

    Args:
        chat_manager: The chat manager instance
    """
    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Initialize messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add the user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for chunk in chat_manager.send_message(prompt, stream=True):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            if full_response.strip():
                audio_file = text_to_speech(full_response)
                st.audio(audio_file, format="audio/wav")

        # Add assistant response to UI
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def render_message(role: str, content: str):
    """
    Render a single message.

    Args:
        role: The role of the message sender (user/assistant)
        content: The message content
    """
    with st.chat_message(role):
        st.markdown(content)
