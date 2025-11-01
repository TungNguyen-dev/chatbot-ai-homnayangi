import logging
from typing import List, Optional, Any

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

# ------------------------------------------------------
# Lazy resources
# ------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_ingredient_collection: Optional[Any] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_ingredient_collection():
    """Retrieve or initialize the Chroma ingredient collection."""
    global _chroma_client, _ingredient_collection
    if _ingredient_collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _ingredient_collection = _chroma_client.get_or_create_collection(name="ingredients")

        # Seed minimal ingredient KB if empty
        if _ingredient_collection.count() == 0:
            base_ingredients = [
                "bánh mì", "bánh tráng", "bắp", "bắp cải", "bạch tuộc", "bí đỏ", "bí xanh", "bơ",
                "bún", "cà chua", "cà rốt", "cà tím", "cải chíp", "cải thìa", "chanh", "dầu",
                "dầu hào", "dưa leo", "đậu cô ve", "đậu hũ", "đậu que", "đường", "gạo", "gừng",
                "hàu", "hành", "hến", "húng quế", "khoai lang", "khoai tây", "mì", "miến",
                "mộc nhĩ",
                "mực", "muối", "nấm", "nước mắm", "ngao", "ngò gai", "ngô", "phô mai", "phở",
                "rau cải", "rau chân vịt", "rau diếp", "rau mồng tơi", "rau muống", "rau mùi",
                "rau ngót", "rau xà lách", "sả", "sữa", "sữa chua", "su su", "tắc", "thịt bò",
                "thịt cừu", "thịt dê", "thịt gà", "thịt heo", "thịt lợn", "thịt ngan", "thịt vịt",
                "tiêu", "tỏi", "tôm", "trứng", "xì dầu",
            ]
            model = get_embedding_model()
            embeddings = model.encode(base_ingredients).tolist()
            _ingredient_collection.add(
                ids=[f"ing_{i}" for i in range(len(base_ingredients))],
                documents=base_ingredients,
                embeddings=embeddings,
            )
    return _ingredient_collection


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _retrieve_similar_ingredients(raw_ingredients: List[str], top_k: int = 2) -> List[str]:
    """Retrieve top-k similar ingredients using Chroma semantic search."""
    unique_raw = _dedupe_preserve_order(raw_ingredients)
    if not unique_raw:
        return []

    model = get_embedding_model()
    collection = get_ingredient_collection()
    embeddings = model.encode(unique_raw).tolist()

    validated: List[str] = []
    for emb in embeddings:
        try:
            result = collection.query(query_embeddings=[emb], n_results=top_k)
            docs = result.get("documents") or []
            if docs and docs[0]:
                validated.extend(docs[0])
        except Exception as e:
            logging.warning(f"Chroma query failed: {e}")
    return _dedupe_preserve_order(validated)


# ------------------------------------------------------
# Main RAG Ingredient Extraction
# ------------------------------------------------------
def detect_ingredients(conversation_history: str, refine: bool = True) -> str:
    """
    Extracts and validates food ingredients mentioned in conversation text.
    Hybrid RAG pipeline:
      1. LLM extraction
      2. Vector validation (Chroma)
      3. Optional refinement (LLM)
    """
    if not conversation_history or not isinstance(conversation_history, str):
        return ""

    client = OpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
    )

    # --- Step 1: LLM Extraction ---
    system_prompt = (
        "You are a Vietnamese culinary assistant specialized in ingredient extraction. "
        "Extract only food ingredient names from the text. Output must be a concise, "
        "comma-separated list of ingredient names in singular form, without explanations."
    )

    user_prompt = f"""
    # === Vietnamese Ingredient Extraction ===
    ## Objective:
    Identify all valid food ingredients from the user's conversation that are commonly used
    in Vietnamese or Asian cooking.

    ## Rules:
    - Output only ingredient names (Vietnamese).
    - Use a comma-separated list (', ').
    - Do NOT include explanations, titles, or additional words.
    - Use singular, common names.
    - Example: "thịt gà, tỏi, ớt"

    ## Input:
    {conversation_history}

    ## Expected Output:
    A single line with only ingredient names, separated by commas.
    """

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )
        content = (response.choices[0].message.content or "").strip()
        raw_ingredients = [i.strip().lower() for i in content.split(",") if i.strip()]
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        return ""

    if not raw_ingredients:
        return ""

    logging.info(f"Raw extracted ingredients: {raw_ingredients}")

    # --- Step 2: Vector Validation (RAG) ---
    validated = _retrieve_similar_ingredients(raw_ingredients)
    detected = validated or raw_ingredients
    logging.info(f"Validated ingredients: {detected}")

    # --- Step 3: Optional LLM Refinement ---
    if refine:
        refine_prompt = f"""
        You are a Vietnamese food assistant.
        Refine and normalize this list of ingredients for proper Vietnamese naming.
        Input: {', '.join(detected)}
        Return only ingredient names in Vietnamese, comma-separated, no extra words.
        """
        try:
            refine_resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.2,
                max_tokens=settings.OPENAI_MAX_TOKENS,
            )
            text = (refine_resp.choices[0].message.content or "").strip()
            refined = [i.strip().lower() for i in text.split(",") if i.strip()]
            return ", ".join(_dedupe_preserve_order(refined))
        except Exception as e:
            logging.warning(f"Refinement failed: {e}")

    return ", ".join(detected)
