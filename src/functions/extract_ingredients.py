"""
Enhanced ingredient extraction module using hybrid RAG approach.
- Step 1: Extract candidate ingredients via OpenAI LLM
- Step 2: Validate and enrich using ChromaDB vector retrieval
- Step 3: Optionally refine/normalize via OpenAI LLM

Optimizations:
- Lazy-load heavy models and vector store on first use to speed up import/startup.
- Batch-encode text for embeddings to reduce redundant compute.
- Preserve input order while de-duplicating results.
- Safer optional LLM refinement and better edge-case handling.
"""

from typing import Any, Iterable, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------
# Function Definition
# ------------------------------------------------------
DEFINITION = {
    "type": "function",
    "function": {
        "name": "extract_ingredients",
        "description": (
            "Extracts and validates food ingredients from user text using OpenAI LLM extraction, "
            "vector retrieval (ChromaDB), and optional LLM refinement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "User text potentially containing food ingredients"},
                "refine": {"type": "boolean",
                           "description": "Whether to refine extracted ingredients via LLM"},
            },
            "required": ["query"],
        },
    },
}

# ------------------------------------------------------
# Lazy getters for heavy resources
# ------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[Any] = None
_ingredient_collection: Optional[Any] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_ingredient_collection():
    """Create or retrieve the Chroma collection and seed a small KB on first use."""
    global _chroma_client, _ingredient_collection
    if _ingredient_collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _ingredient_collection = _chroma_client.get_or_create_collection(name="ingredients")
        # Seed with a small base if empty
        if _ingredient_collection.count() == 0:
            base_ingredients = [
                "chicken", "beef", "salmon", "garlic", "onion", "tomato", "spinach",
                "avocado", "egg", "milk", "butter", "olive oil", "salt", "pepper",
                "broccoli", "rice", "pork", "shrimp", "tofu",
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

def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        key = x.lower() if isinstance(x, str) else x
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _retrieve_similar_ingredients(raw_ingredients: List[str], top_k: int = 2) -> List[str]:
    """Retrieve top-k similar known ingredient names for each raw item.

    Optimizations:
    - Batch-encode all unique inputs with SentenceTransformer once.
    - Preserve the order and uniqueness of the final list.
    """
    unique_raw = _dedupe_preserve_order(
        [s.strip().lower() for s in raw_ingredients if s and s.strip()])
    if not unique_raw:
        return []

    model = get_embedding_model()
    collection = get_ingredient_collection()

    # Batch encode
    embeddings = model.encode(unique_raw).tolist()

    validated: List[str] = []
    for emb in embeddings:
        # Chroma expects a list of query embeddings
        result = collection.query(query_embeddings=[emb], n_results=top_k)
        docs = result.get("documents") or []
        if docs and docs[0]:
            validated.extend(docs[0])

    return _dedupe_preserve_order([v.strip().lower() for v in validated if v and v.strip()])


# ------------------------------------------------------
# Main handler
# ------------------------------------------------------

def handle(llm_client, args: dict) -> List[str]:
    """
    Extract ingredient names using hybrid NER + vector retrieval + LLM refinement.
    """
    query: str = args.get("query", "")
    refine: bool = args.get("refine", True)

    if not query or not isinstance(query, str):
        return []

    # --- Step 1: Extract candidate ingredients via OpenAI ---
    raw_ingredients: List[str] = []
    if llm_client is not None:
        system_prompt = (
            "You are an assistant that extracts food ingredients from text. "
            "Return only a comma-separated list of ingredient names, no extra words. "
            "Use singular, common names (e.g., 'tomato', 'olive oil')."
        )
        user_prompt = (
                "Extract the ingredients mentioned in the following text. "
                "If none, return an empty string.\n\n" + query
        )
        try:
            resp = llm_client._chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = (resp.choices[0].message.content or "").strip()
            raw_ingredients = [s.strip().lower() for s in content.split(",") if s.strip()]
            raw_ingredients = _dedupe_preserve_order(raw_ingredients)
        except Exception:
            raw_ingredients = []

    if not raw_ingredients:
        return []
    print(f"Raw ingredients: {', '.join(raw_ingredients)}")

    # --- Step 2: Vector-based validation (ChromaDB RAG step) ---
    validated_ingredients = _retrieve_similar_ingredients(raw_ingredients)
    print(f"Detected ingredients: {', '.join(validated_ingredients)}")

    # --- Step 3: Optional LLM refinement ---
    if refine and llm_client is not None:
        detected = validated_ingredients if validated_ingredients else raw_ingredients
        print(f"Detected ingredients (refined): {', '.join(detected)}")
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
            f"User query: {query}\n\n"
            "Return only the ingredient names, separated by commas."
        )
        try:
            response = llm_client._chat_completion(messages=[{"role": "user", "content": prompt}])
            text = (response.choices[0].message.content or "").strip()
            print(f"Refined ingredients: {text}")
            final = [i.strip().lower() for i in text.split(",") if i.strip()]
            print(f"Refined ingredients (deduped): {', '.join(final)}")
            return _dedupe_preserve_order(final)
        except (RuntimeError, AttributeError, IndexError, ValueError):
            # Fallback to non-refined if the LLM call fails or a response format is unexpected
            pass

    return validated_ingredients or raw_ingredients
