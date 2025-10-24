"""
Prompt builder that combines system prompts, user prompts, and context.
"""

from typing import List, Dict, Optional

from src.utils.file_loader import load_prompt


class PromptBuilder:
    """Builds prompts by combining system and user prompts with context."""

    def __init__(self):
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> str:
        """Load and combine system prompts."""
        chatbot_role = load_prompt("system_prompts/chatbot_role.txt")
        persona = load_prompt("system_prompts/persona.txt")

        return f"{chatbot_role}\n\n{persona}"

    def build_system_message(
            self, additional_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build the system message with optional additional context.

        Args:
            additional_context: Optional additional context to include

        Returns:
            System message dictionary
        """
        content = self.system_prompts

        if additional_context:
            content = f"{content}\n\nAdditional Context:\n{additional_context}"

        return {"role": "system", "content": content}

    def build_messages(
            self,
            conversation_history: List[Dict[str, str]],
            additional_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build complete message list for the LLM.

        Args:
            conversation_history: List of previous messages
            additional_context: Optional additional context

        Returns:
            Complete list of messages for the LLM
        """
        messages = [self.build_system_message(additional_context)]
        messages.extend(conversation_history)

        return messages

    def load_user_prompt_template(self, template_name: str) -> str:
        """
        Load a user prompt template.

        Args:
            template_name: Name of the template file (e.g., 'faq.txt')

        Returns:
            Template content
        """
        return load_prompt(f"user_prompts/{template_name}")
