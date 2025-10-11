"""
Configuration settings for the chatbot application.
Store API keys, model configurations, and constants here.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

# Application Configuration
APP_TITLE = "Hôm nay ăn gì?"
APP_ICON = "🤖"
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

# Paths
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
SYSTEM_PROMPTS_DIR = os.path.join(PROMPTS_DIR, "system_prompts")
USER_PROMPTS_DIR = os.path.join(PROMPTS_DIR, "user_prompts")

# Memory Configuration
USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "false").lower() == "true"
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
