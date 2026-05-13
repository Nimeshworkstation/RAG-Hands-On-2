from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from data_loader import load_all_documents
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size=2000,
        chunk_overlap=200,
        upload_types=[],
    ):
        self.model_name = model_name
        self.model = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._load_model()
        self.upload_types = upload_types

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def chunk_documents(self, document: List[Document]):
        total_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents=document)
        total_chunks.extend(chunks)
        print(
            f"Splitted {len(document)} Documents into {len(total_chunks)} chunks ✓ \n"
        )
        return total_chunks

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not found")
        print(f"Generating embeddings for {len(texts)} text(s)...\n\n")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"\n✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_documents(self, path, file_types=[]):
        print("\n📥 Step 1/3: Loading documents...")

        if not file_types:
            print("No file type provided to process... ")
            return None

        documents = load_all_documents(path, file_types)

        if not documents:
            print("\nNo documents loaded..")
            return None

        print("\n✂️  Step 2/3: Chunking documents...")
        chunks = self.chunk_documents(documents)

        if not chunks:
            print("\nNo chunks Created..")
            return None

        print("\n🧮 Step 3/3: Generating embeddings...")
        chunk_text = [chunk.page_content for chunk in chunks]
        embeddings = self.generate_embedding(chunk_text)
        if len(chunks) != len(embeddings):
            print(
                f"\nError: chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})"
            )
            return None
        return chunks, embeddings


def main():
    em = EmbeddingManager()
    em.embed_documents(path="../data", file_types=["txt", "pdf"])


if __name__ == "__main__":
    main()
