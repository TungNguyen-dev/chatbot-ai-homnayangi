"""
Configuration settings for the chatbot application.
Store API keys, model configurations, and constants here.
"""

import os
from dotenv import load_dotenv

# Resolve path to the project root (two levels above this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# Define the .env file path
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load environment variables
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"✅ Environment loaded from: {ENV_PATH}")
else:
    print(f"⚠️  Warning: .env file not found at {ENV_PATH}")

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
