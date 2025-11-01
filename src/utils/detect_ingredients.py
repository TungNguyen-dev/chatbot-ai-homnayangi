from openai import OpenAI

from src.config.settings import settings


def detect_ingredients(messages):
    system_prompt = (
        "You are an assistant that extracts food ingredients from text. "
        "Return only a comma-separated list of ingredient names, no extra words. "
        "Use singular, common names (e.g., 'tomato', 'olive oil')."
    )
    prompt = (
        "# === Vietnamese Ingredient Extraction Prompt ===\n"
        "# Description:\n"
        "#   Given a free-form user query, identify all valid food ingredients "
        "#   that could realistically be used in Vietnamese or Asian cooking.\n"
        "# Goal:\n"
        "#   Return only ingredient names (in Vietnamese), separated by commas.\n"
        "# Output Format:\n"
        "#   - A single line string of ingredients, separated by ', '.\n"
        "#   - Do NOT include explanations, titles, or extra words.\n"
        "#   - Example output: 'gà, tỏi, ớt'\n"
        "# ==============================================================\n\n"
        "You are a Vietnamese food ingredient extraction assistant.\n"
        "Given a user's message, extract and return only the list of valid ingredient names.\n\n"
        f"User messages: {messages}\n\n"
        "Return only the ingredient names, separated by commas."
    )

    client = OpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
    )
    content = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )
    return content.choices[0].message.content
