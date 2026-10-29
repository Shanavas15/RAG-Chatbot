import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        print(f"Loading document: {self.file_path}...")
        try:
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF '{self.file_path}': {e}")
        print(f"Loaded {len(docs)} pages.")
        return docs