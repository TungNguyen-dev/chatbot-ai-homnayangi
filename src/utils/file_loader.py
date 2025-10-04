"""
File loading utilities for prompts and configuration files.
"""

import os
from typing import Optional
from src.config.settings import PROMPTS_DIR


def load_prompt(filename: str) -> str:
    """
    Load a prompt file from the prompts directory.

    Args:
        filename: Name of the prompt file (can include subdirectory)

    Returns:
        Content of the prompt file
    """
    file_path = os.path.join(PROMPTS_DIR, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Prompt file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading prompt file {file_path}: {e}")
        return ""


def load_file(file_path: str, default: Optional[str] = None) -> str:
    """
    Load any text file.

    Args:
        file_path: Path to the file
        default: Default content if file not found

    Returns:
        Content of the file or default value
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        if default is not None:
            return default
        print(f"File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return default or ""
