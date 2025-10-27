"""
Configuration settings for the chatbot application.
Store API keys, model configurations, and constants here.
"""

import os

from dotenv import load_dotenv


class Settings:
    """Configuration settings for the chatbot application."""

    def __init__(self):
        # Resolve the path to the project root (two levels above this file)
        self.BASE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )

        # Define the .env file path
        self.ENV_PATH = os.path.join(self.BASE_DIR, ".env")

        # Load environment variables
        self._load_environment()

        # API Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
        self.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

        self.OPENAI_EMBEDDING_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL")
        self.OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")

        # Application Configuration
        self.APP_TITLE = "Hôm nay ăn gì?"
        self.APP_ICON = "🤖"
        self.MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

        # Paths
        self.PROMPTS_DIR = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "prompts"
        )
        self.SYSTEM_PROMPTS_DIR = os.path.join(self.PROMPTS_DIR, "system_prompts")
        self.USER_PROMPTS_DIR = os.path.join(self.PROMPTS_DIR, "user_prompts")

        # Memory Configuration
        self.USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "false").lower() == "true"
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

    def _load_environment(self):
        """Load environment variables from .env file."""
        if os.path.exists(self.ENV_PATH):
            load_dotenv(dotenv_path=self.ENV_PATH)
            print(f"✅ Environment loaded from: {self.ENV_PATH}")
        else:
            print(f"⚠️  Warning: .env file not found at {self.ENV_PATH}")


# Create a singleton instance
settings = Settings()
