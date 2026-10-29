from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        # Uses tiktoken encoder under the hood to split based on token count
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of Document objects into smaller chunk Document objects."""
        print("Chunking documents...")
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} pages.")
        return chunks