import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
from langchain_core.documents import Document

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initializes FAISS index with specified vector dimensionality.
        `all-MiniLM-L6-v2` produces 384-dimensional vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.docs: List[Document] = []

    def build_index(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Populates FAISS index with embedding vectors and aligns metadata.
        """
        embeddings_float32 = np.array(embeddings).astype('float32')        
        self.index.add(embeddings_float32)
        self.docs.extend(documents)
        print(f"FAISS Index built with {self.index.ntotal} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Performs semantic similarity search for a query vector.
        Returns a list of tuples: (Document Chunk, Distance Score).
        Lower distance score indicates higher similarity.
        """
        query_vector_32 = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector_32, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.docs):  # Valid index check
                results.append((self.docs[idx], float(dist)))
        return results

    def save_index(self, folder_path: str = "vector_store"):
        """
        Persists the FAISS index binary and document metadata pickle to disk.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        index_file = os.path.join(folder_path, "faiss.index")
        docs_file = os.path.join(folder_path, "docs.pkl")

        faiss.write_index(self.index, index_file)
        with open(docs_file, "wb") as f:
            pickle.dump(self.docs, f)

        print(f"FAISS Index & Metadata saved successfully to folder: '{folder_path}'")

    def load_index(self, folder_path: str = "vector_store"):
        """
        Loads FAISS index binary and document metadata from disk.
        """
        index_file = os.path.join(folder_path, "faiss.index")
        docs_file = os.path.join(folder_path, "docs.pkl")

        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(f"Index files missing in directory: '{folder_path}'")

        self.index = faiss.read_index(index_file)
        with open(docs_file, "rb") as f:
            self.docs = pickle.load(f)

        print(f"Loaded FAISS Index with {self.index.ntotal} vectors from '{folder_path}'.")