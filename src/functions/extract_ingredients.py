"""
Enhanced ingredient extraction module using hybrid RAG approach.
- Step 1: Extract candidate ingredients via Hugging Face NER model
- Step 2: Validate and enrich using ChromaDB vector retrieval
- Step 3: Optionally refine/normalize via OpenAI LLM
"""

from typing import List

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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
# Load Hugging Face models
# ------------------------------------------------------
_ner_pipeline = pipeline(
    "token-classification",
    model="fran-martinez/food-ner",
    aggregation_strategy="simple",
)

_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------
# Initialize ChromaDB (can be persistent for caching)
# ------------------------------------------------------
_chroma_client = chromadb.PersistentClient(path="./chroma_store")
_ingredient_collection = _chroma_client.get_or_create_collection(name="ingredients")

# Example ingredient knowledge base (can be replaced with a larger dataset)
if _ingredient_collection.count() == 0:
    base_ingredients = [
        "chicken", "beef", "salmon", "garlic", "onion", "tomato", "spinach",
        "avocado", "egg", "milk", "butter", "olive oil", "salt", "pepper",
        "broccoli", "rice", "pork", "shrimp", "tofu"
    ]
    embeddings = _embedding_model.encode(base_ingredients).tolist()
    _ingredient_collection.add(
        ids=[f"ing_{i}" for i in range(len(base_ingredients))],
        documents=base_ingredients,
        embeddings=embeddings
    )


# ------------------------------------------------------
# Helper: Retrieve similar ingredient names
# ------------------------------------------------------
def _retrieve_similar_ingredients(raw_ingredients: List[str], top_k: int = 2) -> List[str]:
    validated = []
    for item in raw_ingredients:
        embedding = _embedding_model.encode([item]).tolist()
        result = _ingredient_collection.query(query_embeddings=embedding, n_results=top_k)
        candidates = result["documents"][0]
        if candidates:
            validated.extend(candidates)
    return list(set(validated))


# ------------------------------------------------------
# Main handler
# ------------------------------------------------------
def handle(llm_client, args: dict) -> List[str]:
    """
    Extract ingredient names using hybrid NER + vector retrieval + LLM refinement.
    """
    query: str = args.get("query", "")
    refine: bool = args.get("refine", True)

    # --- Step 1: NER extraction ---
    entities = _ner_pipeline(query)
    raw_ingredients = [e["word"].lower() for e in entities if e["entity_group"] == "FOOD"]
    raw_ingredients = list(set(raw_ingredients))

    if not raw_ingredients:
        return []

    # --- Step 2: Vector-based validation (ChromaDB RAG step) ---
    validated_ingredients = _retrieve_similar_ingredients(raw_ingredients)

    # --- Step 3: Optional LLM refinement ---
    if refine:
        prompt = (
            "You are a food ingredient normalization assistant.\n"
            "Given a user query and a list of detected ingredients, "
            "produce a clean, comma-separated list of valid ingredient names.\n\n"
            f"User query: {query}\n"
            f"Detected ingredients: {', '.join(validated_ingredients)}\n\n"
            "Response:"
        )

        response = llm_client._chat_completion(messages=[{"role": "user", "content": prompt}])
        text = response.choices[0].message.content.strip()
        final = [i.strip().lower() for i in text.split(",") if i.strip()]
        return list(set(final))

    return validated_ingredients
