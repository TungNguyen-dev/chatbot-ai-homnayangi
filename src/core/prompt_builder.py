"""
Prompt builder that combines system prompts, user prompts, and context.
"""
import os
from typing import List, Dict, Optional

from src.utils.file_loader import load_prompt


class PromptBuilder:
    """Builds prompts by combining system and user prompts with context."""

    def __init__(self):
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> str:
        """
        Load and combine all system prompt files into a single formatted string.
        Ensures the final result is always a valid string.
        Automatically skips missing or empty files.
        """

        prompt_files = [
            "system_prompts/chatbot_role.txt",
            "system_prompts/persona.txt",
            "system_prompts/nhat_ky_an_uong1.txt",
            "system_prompts/thoi_quen_an_uong1.txt",
            "system_prompts/vietnamese_dishes_prompt.txt",
        ]

        sections = []

        for path in prompt_files:
            try:
                content = load_prompt(path)
                # Ensure content is string
                if not isinstance(content, str):
                    content = str(content or "")
                content = content.strip()
                if content:
                    section_title = os.path.splitext(os.path.basename(path))[0].replace("_",
                                                                                        " ").title()
                    sections.append(f"### {section_title}\n{content}")
                else:
                    print(f"[Warning] Empty prompt file skipped: {path}")
            except FileNotFoundError:
                print(f"[Warning] Missing prompt file: {path}")
            except Exception as e:
                print(f"[Error] Failed to load prompt '{path}': {e}")

        if not sections:
            raise RuntimeError("No valid system prompts loaded.")

        # Combine all into a single string
        combined_prompt = "\n\n".join(sections)

        # Final safe return — always a string
        result = (
            "# === SYSTEM PROMPT COLLECTION ===\n"
            "You are provided with multiple context definitions below.\n"
            "Use them collectively to understand your role, persona, and domain knowledge.\n\n"
            f"{combined_prompt.strip()}"
        )

        return str(result)

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
