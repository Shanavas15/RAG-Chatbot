from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder model.
        This model outputs a raw relevance score between a query and a passage.
        """
        print(f"Loading Cross-Encoder model: '{model_name}'...")
        self.model = CrossEncoder(model_name)
        print("Cross-Encoder loaded successfully!")

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Takes a pool of retrieved documents and re-scores them against the query.
        """
        if not documents:
            return []

        # 1. Create pairs of [Query, Document Text]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 2. Predict relevance scores (this is the heavy AI lifting)
        scores = self.model.predict(pairs)
        
        # 3. Zip documents with their new scores
        scored_docs = list(zip(documents, scores))
        
        # 4. Sort by highest score first
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Extract and return the final Top K documents
        return [doc for doc, _score in scored_docs[:top_k]]