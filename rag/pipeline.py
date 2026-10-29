import os
from typing import List, Tuple

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

from rag.bm25_store import BM25Store
from rag.embeddings import EmbeddingModel
from rag.faiss_store import FAISSVectorStore
from rag.generator import NO_ANSWER, LLMError, RAGGenerator
from rag.hybrid_retriever import HybridRetriever
from rag.loader import PDFLoader
from rag.reranker import CrossEncoderReRanker
from rag.splitter import DocumentSplitter


class EnterpriseRAGPipeline:
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = vector_store_path
        self.embedder = EmbeddingModel()

        self.faiss_store = FAISSVectorStore(dimension=384)
        if os.path.exists(vector_store_path):
            self.faiss_store.load_index(vector_store_path)

        self.bm25_store = BM25Store()
        if os.path.exists(vector_store_path):
            self.bm25_store.load_index(vector_store_path)

        self.hybrid = HybridRetriever(self.faiss_store, self.bm25_store, self.embedder)
        self.reranker = CrossEncoderReRanker()

        self.generator = RAGGenerator()

        print("Initializing Web Search Agent...")
        self.web_search = DuckDuckGoSearchResults()

    def ingest_new_document(self, file_path: str) -> int:
        print(f"Ingesting new document: {file_path}")
        loader = PDFLoader(file_path)
        documents = loader.load()
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
        new_chunks = splitter.split_documents(documents)

        chunk_texts = [chunk.page_content for chunk in new_chunks]
        new_vectors = self.embedder.embed_documents(chunk_texts)

        self.faiss_store.build_index(new_vectors, new_chunks)
        self.faiss_store.save_index(self.vector_store_path)

        all_documents = self.faiss_store.docs
        self.bm25_store.build_index(all_documents)
        self.bm25_store.save_index(self.vector_store_path)

        return len(new_chunks)

    def ask(self, query: str, chat_history: str = "", top_k: int = 3) -> Tuple[str, List, str]:
        """Returns (answer, sources, source_type), where source_type is 'local', 'web', or 'error'."""
        if not self.faiss_store.docs:
            return "Please upload a document first.", [], "error"

        hybrid_pool = self.hybrid.search(query, top_k=6, pool_size=10)
        final_docs = self.reranker.rerank(query, hybrid_pool, top_k=top_k)

        try:
            answer = self.generator.generate_answer(query, final_docs, chat_history)
        except LLMError as e:
            return f"The language model request failed: {e}", [], "error"

        if answer != NO_ANSWER and NO_ANSWER not in answer:
            return answer, final_docs, "local"

        standalone_query = self.generator.rewrite_query(query, chat_history)
        web_results = self.web_search.run(standalone_query)

        try:
            web_answer = self.generator.generate_web_answer(query, web_results)
        except LLMError as e:
            return f"The language model request failed: {e}", [], "error"

        return web_answer, web_results, "web"