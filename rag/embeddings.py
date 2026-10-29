from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
        texts, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
        query, 
        convert_to_numpy=True,
        normalize_embeddings=True
        )
        return embedding