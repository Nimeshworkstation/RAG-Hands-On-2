import chromadb
import os
import uuid
import hashlib
from langchain_core.documents import Document
from typing import List
import numpy as np


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "pdf-documents",
        persist_directory: str = "store/vector_store/chromadb/",
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
                metadata={
                    "description": "PDF document embedding for RAG",
                    "hnsw:space": "cosine",
                },
            )
            print(self.collection)
            print(self.collection.metadata)
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error occured {e}")
            raise

    def create_chunk_id(self, chunk, index):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        text = chunk.page_content
        raw = f"{source}|{page}|{text}|{index}".encode("utf-8")
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

        for index, (chunk, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(self.create_chunk_id(chunk, index))
            documents_text.append(chunk.page_content)
            embeddings_list.append(embedding.tolist())
            metadata = dict(chunk.metadata)
            metadata["doc_index"] = index
            metadata["content_length"] = len(chunk.page_content)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        print(f"[INFO] Stored {len(documents)} chunks in ChromaDB")

    def retrieve_documents(
        self, query_embeddings: np.ndarray, top_k: int = 3, score_threshold: float = 0.0
    ):
        retrieved_docs = []
        result = self.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if result.get("documents") and result.get("documents")[0]:
            documents = result.get("documents")[0]
            metadatas = result.get("metadatas")[0]
            distances = result.get("distances")[0]
            ids = result.get("ids")[0]

            for index, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):

                similiarity_score = 1 - distance
                if similiarity_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similiarity_score,
                            "rank": index + 1,
                        }
                    )
            print(f"Retrieved {len(retrieved_docs)} docuemnt (after filtering)")
        else:
            print(
                f"No collection named '{self.collection_name}' was found at "
                f"'{self.persist_directory}'. Check the collection before querying."
            )

        return retrieved_docs


def main():
    vectorstore = ChromaVectorStore()


if __name__ == "__main__":
    main()
