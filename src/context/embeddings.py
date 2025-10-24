"""
Embeddings module for vector database operations.
Optional: Used if you want to implement semantic search or long-term memory.
"""

from typing import List, Optional
import chromadb
from src.config.settings import settings


class EmbeddingsManager:
    """Manages embeddings and vector database operations."""

    def __init__(self):
        self.enabled = settings.USE_VECTOR_DB
        self.client = None
        self.collection = None

        if self.enabled:
            self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            self.collection = self.client.get_or_create_collection(
                name="conversation_history"
            )
        except Exception as e:
            print(f"Failed to initialize vector DB: {e}")
            self.enabled = False

    def add_text(
            self, text: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None
    ):
        """Add text to the vector database."""
        if not self.enabled or not self.collection:
            return

        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id or str(hash(text))],
            )
        except Exception as e:
            print(f"Failed to add text to vector DB: {e}")

    def search_similar(self, query: str, n_results: int = 5) -> List[str]:
        """Search for similar texts in the database."""
        if not self.enabled or not self.collection:
            return []

        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)

            # Ensure the query actually returned a result
            if not results or "documents" not in results or not results["documents"]:
                print("No documents found for the given query.")
                return []

            # Safely extract the documents list
            return results["documents"][0] if results["documents"][0] else []

        except Exception as e:
            print(f"Failed to search vector DB: {e}")
            return []
