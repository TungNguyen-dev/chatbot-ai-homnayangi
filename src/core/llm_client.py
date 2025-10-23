"""
LLM client wrapper for OpenAI API.
"""

from typing import Optional, Generator, Union
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from src.config.settings import settings
from src.core.function_dispatcher import FUNCTION_DEFINITIONS, FunctionDispatcher


class LLMClient:
    """Wrapper around OpenAI API for LLM interactions only."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        )
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.dispatcher = FunctionDispatcher(self)

    # ------------------------------
    # Generic LLM Calls
    # ------------------------------

    def _chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        """
        Low-level API call to OpenAI chat completion (OpenAI ≥1.x).
        Returns a ChatCompletion or a streaming iterator.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_completion_tokens=max_tokens or self.max_tokens,  # ✅ Updated param
                stream=stream,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI chat completion failed: {e}")

    # ------------------------------
    # High-Level Generation Methods
    # ------------------------------

    def generate_response(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a full, synchronous response from the model.
        """
        try:
            completion = self._chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            # ✅ Explicitly check type for Pyright
            if isinstance(completion, ChatCompletion):
                return completion.choices[0].message.content or ""
            else:
                return "Error: Unexpected streaming object returned."

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream LLM responses and handle function/tool calls via FunctionDispatcher.
        """
        try:
            stream = self._chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=FUNCTION_DEFINITIONS,
                tool_choice="auto",
            )

            # ✅ Type guard ensures only Stream is handled
            if isinstance(stream, Stream):
                yield from self.dispatcher.handle_stream(stream)
            else:
                yield "Error: Expected streaming response but got full completion."

        except Exception as e:
            yield f"Error: {str(e)}"

