from typing import List, Optional
import os
import json
import chromadb
from openai import OpenAI
from src.config.settings import settings


class EmbeddingsManager:
    """Manages embeddings and vector database operations."""

    def __init__(self):
        self.enabled = settings.USE_VECTOR_DB
        self.client = None
        self.collection = None
        self.client_openai = OpenAI(
            base_url=settings.OPENAI_EMBEDDING_BASE_URL,
            api_key=settings.OPENAI_EMBEDDING_API_KEY,
        )

        if self.enabled:
            self._initialize_db()
            self._load_initial_data()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            # Không cần embedding_function vì chúng ta sẽ cung cấp embedding sẵn
            self.collection = self.client.get_or_create_collection(
                name="conversation_history",
                embedding_function=None,
            )
        except Exception as e:
            print(f"Failed to initialize vector DB: {e}")
            self.enabled = False

    def _load_initial_data(self):
        """Load initial food data into vector DB."""
        print("🧠 Đang nạp dữ liệu món ăn mặc định...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "favourites", "favourites.json")

        with open(file_path, "r", encoding="utf-8") as f:
            foods = json.load(f)

        for food in foods:
            # Tạo embedding bằng OpenAI
            embedding = self.client_openai.embeddings.create(
                model="text-embedding-3-small",  # 384 chiều
                input=food["desc"]
            ).data[0].embedding

            # Add vào collection
            try:
                self.collection.add(
                    ids=[food["id"]],
                    embeddings=[embedding],  # ✅ bắt buộc nếu không có embedding_function
                    documents=[food["desc"]],
                    metadatas=[{
                        "name": food["name"],
                        "tags": ", ".join(food["tags"])  # ✅ chuyển list -> string
                    }]
                )
            except Exception as e:
                print(f"Failed to add food {food['name']}: {e}")

        print("✅ Đã preload dữ liệu món ăn vào vector DB.")

    def add_text(
        self, text: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None
    ):
        """Add text to the vector database."""
        if not self.enabled or not self.collection:
            return

        try:
            # Tạo embedding trước khi thêm
            embedding = self.client_openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            self.collection.add(
                documents=[text],
                embeddings=[embedding],  # ✅ bắt buộc
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
            # Tính embedding query trước
            query_embedding = self.client_openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            ).data[0].embedding

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            if not results or "documents" not in results or not results["documents"]:
                return []

            return results["documents"][0] if results["documents"][0] else []

        except Exception as e:
            print(f"Failed to search vector DB: {e}")
            return []
