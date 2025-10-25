"""
High-level chat manager that orchestrates all components.
"""

from typing import Generator, Union, cast

from openai.types.chat import ChatCompletionMessageParam

from src.context.embeddings import EmbeddingsManager
from src.context.memory_manager import MemoryManager
from src.core.llm_client import LLMClient
from src.core.prompt_builder import PromptBuilder


class ChatManager:
    """Manages chat interactions and coordinates all components."""

    def __init__(self):
        self.memory = MemoryManager()
        self.embeddings = EmbeddingsManager()
        self.llm = LLMClient()
        self.prompt_builder = PromptBuilder()

    def send_message(self, user_message: str, stream: bool = False) -> Union[
        str, Generator[str, None, None]]:
        """
        Send a user message and get a response.

        Args:
            user_message: The user's message
            stream: Whether to stream the response

        Returns:
            The assistant's response (str) or a streaming generator of response chunks
        """
        # Add the user message to memory
        self.memory.add_message("user", user_message)

        # Optionally add to vector DB for long-term memory
        if self.embeddings.enabled:
            self.embeddings.add_text(user_message, metadata={"role": "user"})

        # 🆕 3️⃣ Truy vấn vector DB xem có món nào phù hợp với câu hỏi hoặc sở thích không
        similar_items = []
        if self.embeddings.enabled:
            similar_items = self.embeddings.search_similar(user_message, n_results=3)

        # 🆕 4️⃣ Nếu có kết quả, tạo đoạn context để AI dùng
        context_info = ""
        if similar_items:
            context_info = (
                "Dưới đây là một vài món ăn bạn có thể cân nhắc (ưu tiên theo sở thích của người dùng):\n"
                + "\n".join(f"- {item}" for item in similar_items)
            )
        print(context_info)

        # Build messages for LLM
        messages = self.prompt_builder.build_messages(self.memory.get_messages())

        # Thêm 1 tin nhắn “system” mới chứa gợi ý món ăn gần nhất
        if context_info:
            messages.insert(1, {"role": "system", "content": context_info})

        # ✅ Explicitly cast to maintain type safety
        typed_messages = cast(list[ChatCompletionMessageParam], cast(object, messages))

        # Generate response
        if stream:
            return self._generate_streaming_response(typed_messages)
        else:
            response = self.llm.generate_response(typed_messages)
            self.memory.add_message("assistant", response)
            return response

    def _generate_streaming_response(
            self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.
        """
        full_response = ""

        for chunk in self.llm.generate_response_stream(messages):
            if chunk:
                yield chunk
                full_response += chunk

        # Add a complete response to memory after streaming
        self.memory.add_message("assistant", full_response)

    def get_conversation_history(self):
        """Get the current conversation history."""
        return self.memory.get_messages()

    def clear_conversation(self):
        """Clear the conversation history."""
        self.memory.clear()

    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        return self.memory.get_context_summary()
