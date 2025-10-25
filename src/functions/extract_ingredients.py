"""
Enhanced ingredient extraction module using hybrid RAG approach.
- Step 1: Extract candidate ingredients via Hugging Face NER model
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
from transformers import pipeline
from transformers.pipelines import TokenClassificationPipeline

# ------------------------------------------------------
# Function Definition
# ------------------------------------------------------
DEFINITION = {
    "type": "function",
    "function": {
        "name": "extract_ingredients",
        "description": (
            "Extracts and validates food ingredients from user text using NER, "
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
_ner_pipeline: Optional[TokenClassificationPipeline] = None
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[Any] = None
_ingredient_collection: Optional[Any] = None


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = pipeline(
            "token-classification",
            model="fran-martinez/food-ner",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_ingredient_collection():
    """Create or retrieve the Chroma collection and seed a small KB on first use."""
    global _chroma_client, _ingredient_collection
    if _ingredient_collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_store")
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

    # --- Step 1: NER extraction ---
    ner = get_ner_pipeline()
    entities = ner(query)
    raw_ingredients = [
        (e.get("word") or "").strip().lower()
        for e in entities
        if (e.get("entity_group") == "FOOD") and e.get("word")
    ]
    raw_ingredients = _dedupe_preserve_order(raw_ingredients)

    if not raw_ingredients:
        return []

    # --- Step 2: Vector-based validation (ChromaDB RAG step) ---
    validated_ingredients = _retrieve_similar_ingredients(raw_ingredients)

    # --- Step 3: Optional LLM refinement ---
    if refine and llm_client is not None:
        detected = validated_ingredients if validated_ingredients else raw_ingredients
        prompt = (
            "You are a food ingredient normalization assistant.\n"
            "Given a user query and a list of detected ingredients, "
            "produce a clean, comma-separated list of valid ingredient names.\n\n"
            f"User query: {query}\n"
            f"Detected ingredients: {', '.join(detected)}\n\n"
            "Response:"
        )
        try:
            response = llm_client._chat_completion(messages=[{"role": "user", "content": prompt}])
            text = (response.choices[0].message.content or "").strip()
            final = [i.strip().lower() for i in text.split(",") if i.strip()]
            return _dedupe_preserve_order(final)
        except (RuntimeError, AttributeError, IndexError, ValueError):
            # Fallback to non-refined if the LLM call fails or a response format is unexpected
            pass

    return validated_ingredients or raw_ingredients
