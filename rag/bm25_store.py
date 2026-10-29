import os
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class BM25Store:
    def __init__(self):
        """
        Initializes the BM25 keyword search engine.
        Unlike FAISS, BM25 doesn't use neural embeddings. It uses term frequency.
        """
        self.bm25 = None
        self.docs: List[Document] = []

    def build_index(self, documents: List[Document]):
        """
        Tokenizes the documents and builds the BM25 index.
        """
        self.docs = documents
        # Basic tokenization: lowercase and split by spaces. 
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.docs]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 Index built with {len(self.docs)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Searches the BM25 index for the exact keywords in the query.
        Returns a list of tuples: (Document Chunk, BM25 Score).
        Higher score = better keyword match.
        """
        if not self.bm25:
            raise ValueError("BM25 index not built or loaded.")
            
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair documents with their scores and sort descending
        scored_docs = list(zip(self.docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top K results
        return scored_docs[:top_k]

    def save_index(self, folder_path: str = "vector_store"):
        """
        Saves the BM25 index and documents to disk.
        """
        os.makedirs(folder_path, exist_ok=True)
        
        bm25_file = os.path.join(folder_path, "bm25_index.pkl")
        docs_file = os.path.join(folder_path, "bm25_docs.pkl")

        with open(bm25_file, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(docs_file, "wb") as f:
            pickle.dump(self.docs, f)
            
        print(f"BM25 Index saved successfully to '{folder_path}'")

    def load_index(self, folder_path: str = "vector_store"):
        """
        Loads the BM25 index and documents from disk.
        """
        bm25_file = os.path.join(folder_path, "bm25_index.pkl")
        docs_file = os.path.join(folder_path, "bm25_docs.pkl")

        if not os.path.exists(bm25_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(f"BM25 Index files missing in: '{folder_path}'")

        with open(bm25_file, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(docs_file, "rb") as f:
            self.docs = pickle.load(f)
            
        print(f"Loaded BM25 Index with {len(self.docs)} chunks.")