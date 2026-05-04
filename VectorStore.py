import chromadb
import os
import uuid
import hashlib
from langchain_core.documents import Document
from typing import List
import numpy as np


class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf-documents",
        persist_directory: str = "../data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG"},
            )
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error occured {e}")
            raise

    def create_chunk_id(self, chunk):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        text = chunk.page_content
        raw = f"{source}|{page}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise RuntimeError("Something went wrong")
        print(f"Adding {len(documents)} vector")

        # prepare data for chroma db
        ids = []
        documents_text = []
        embeddings_list = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(self.create_chunk_id(chunk))
            documents_text.append(chunk.page_content)
            embeddings_list.append(embedding.tolist())
            metadata = dict(chunk.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(chunk.page_content)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        print(f"[INFO] Stored {len(documents)} chunks in ChromaDB")


def main():
    vectorstore = VectorStore()


if __name__ == "__main__":
    main()
