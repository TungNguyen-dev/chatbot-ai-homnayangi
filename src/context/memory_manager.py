"""
Memory manager for handling conversation context and history.
"""

from typing import List, Dict
from src.config.settings import settings


class MemoryManager:
    """Manages conversation memory and context."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = settings.MAX_CONTEXT_MESSAGES

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        self._trim_messages()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation history."""
        return self.messages.copy()

    def _trim_messages(self):
        """Keep only the most recent messages within the limit."""
        if len(self.messages) > self.max_messages:
            # Keep system messages and trim oldest user/assistant messages
            system_messages = [m for m in self.messages if m["role"] == "system"]
            other_messages = [m for m in self.messages if m["role"] != "system"]

            # Keep most recent messages
            other_messages = other_messages[
                -(self.max_messages - len(system_messages)) :
            ]
            self.messages = system_messages + other_messages

    def clear(self):
        """Clear all conversation history."""
        self.messages = []

    def get_context_summary(self) -> str:
        """Generate a summary of the current context."""
        if not self.messages:
            return "No conversation history."

        return f"Conversation has {len(self.messages)} messages."
