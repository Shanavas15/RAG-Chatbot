from typing import List
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, faiss_store, bm25_store, embedder):
        """
        Initializes the Hybrid Retriever with our three core components.
        """
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, pool_size: int = 15) -> List[Document]:
        """
        Performs hybrid search using Reciprocal Rank Fusion (RRF).
        pool_size: How many results to pull from each index before fusing.
        """
        # 1. Get FAISS (Semantic) Results
        query_vector = self.embedder.embed_query(query)
        faiss_results = self.faiss.search(query_vector, top_k=pool_size)
        
        # 2. Get BM25 (Keyword) Results
        bm25_results = self.bm25.search(query, top_k=pool_size)
        
        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_k = 60 # Standard smoothing constant used by Pinecone/Elasticsearch
        fused_scores = {}
        
        # Helper function to process ranks
        def process_ranks(results_list):
            for rank, (doc, _score) in enumerate(results_list, start=1):
                # We use the text content as a unique ID to merge identical chunks
                doc_id = doc.page_content 
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {'doc': doc, 'score': 0.0}
                fused_scores[doc_id]['score'] += 1.0 / (rrf_k + rank)

        # Process both lists
        process_ranks(faiss_results)
        process_ranks(bm25_results)
        
        # 4. Sort by the new RRF score descending
        reranked = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        # 5. Extract and return the Top K Document objects
        return [item['doc'] for item in reranked[:top_k]]