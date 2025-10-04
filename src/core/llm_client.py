"""
LLM client wrapper for OpenAI API (or other LLM providers).
"""

from typing import List, Dict, Optional
from openai import OpenAI
from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    OPENAI_BASE_URL,
)


class LLMClient:
    """Wrapper around OpenAI API for LLM interactions."""

    def __init__(self):
        self.client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.temperature = OPENAI_TEMPERATURE
        self.max_tokens = OPENAI_MAX_TOKENS

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
            )

            if stream:
                return response

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error: {str(e)}"

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a streaming response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Yields:
            Response chunks as they are generated
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"
