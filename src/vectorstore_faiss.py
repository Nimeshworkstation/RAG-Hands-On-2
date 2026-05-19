import os
import re
import numpy as np
import faiss
import json
from datetime import datetime


class FaissVectorStore:
    def __init__(
        self,
        persist_directory="store/vector_store/faiss/",
        vector_dimension=384,
        faiss_index_name="faiss_index.bin",
        faiss_metadata_name="faiss_metadata.json",
    ):
        self.persist_directory = persist_directory
        self.embedding_dimension = vector_dimension
        self.faiss_index_name = faiss_index_name
        self.index = None
        self.metadata = []
        self.faiss_metadata_name = faiss_metadata_name
        self._initialize_index_metadata()
        self._get_stats()

    def _get_stats(self):
        stats = {
            "total_vectors": self.index.ntotal,
            "total_metadata": len(self.metadata),
            "embedding_dimension": self.embedding_dimension,
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
        }

        print("\nVector Store Stats")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 30)

        return stats

    def _initialize_index_metadata(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.index_path = os.path.join(
                self.persist_directory,
                self.faiss_index_name,
            )
            self.metadata_path = os.path.join(
                self.persist_directory,
                self.faiss_metadata_name,
            )

            if os.path.exists(self.index_path):
                print("Index File  found and Initializing....")
                self.index = faiss.read_index(self.index_path)

            else:
                print("Creating new Index File...")
                self.index = faiss.IndexFlatIP(self.embedding_dimension)

            if (
                os.path.exists(self.metadata_path)
                and os.path.getsize(self.metadata_path) > 0
            ):
                print("Previous Metadata found, Loading....")
                with open(self.metadata_path, "r") as file:
                    self.metadata = json.load(file)

            else:
                print("Creating new Metadata...")
                self.metadata = []

                with open(self.metadata_path, "w") as file:
                    json.dump(self.metadata, file, indent=2)

            if self.index.ntotal != len(self.metadata):
                print(
                    f"⚠️  Warning: Index has {self.index.ntotal} vectors "
                    f"but metadata has {len(self.metadata)} entries"
                )

        except Exception as e:
            print(f"Error occurred during initialization: {e}")
            raise

    def normalize_embeddings(self, embeddings):
        np_embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(np_embeddings)
        return np_embeddings

    def add_to_index(self, normalized_embeddings):
        self.index.add(normalized_embeddings)

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}")

    def create_metadata(self, chunk, index_id):
        metadata = {}
        metadata["id"] = str(index_id)
        metadata["source"] = chunk.metadata.get("source", "")
        metadata["file_name"] = chunk.metadata.get("file_name", "")
        metadata["extension"] = chunk.metadata.get("extension", "")
        metadata["created_at"] = chunk.metadata.get(
            "creationdate", datetime.now().isoformat()
        )
        metadata["content"] = (
            re.sub(r"[ \t]+", " ", chunk.page_content).strip()
            if chunk.page_content
            else ""
        )

        return metadata

    def save_index_metadata(self, chunks):
        if not chunks:
            print("Nothing provided to Save.. Nothing saved")
            return
        chunk_list, embedding_array = chunks
        np_embeddings = self.normalize_embeddings(embedding_array)
        starting_id = self.index.ntotal
        self.add_to_index(np_embeddings)
        self.save_index()
        for idx, chunk in enumerate(chunk_list):
            data = self.create_metadata(chunk=chunk, index_id=(starting_id + idx))
            self.metadata.append(data)

        with open(self.metadata_path, "w") as file:
            json.dump(self.metadata, file, indent=2)

        print(f"✓ Added {len(chunk_list)} new entries to vector store")
        print(f"✓ Total indexes(vectors): {self.index.ntotal}")
        print(f"✓ Total metadata entries: {len(self.metadata)}")

    def retreive_documents(self, query_embeddings, top_k=3):
        retrieved_texts = []
        if not self.index.ntotal:
            print("Nothing to Compare. check the index file ")
            return []

        if len(query_embeddings) == 0:
            print("No vectors for query. check the query ")
            return []

        if top_k > self.index.ntotal:
            print("top_k should be smaller than number of index..")
            return []

        query_np_embeddings = self.normalize_embeddings(query_embeddings)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_np_embeddings, k)

        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]
                retrieved_texts.append(
                    {
                        "similiarity_score": float(score),
                        "id": meta.get("id"),
                        "content": meta.get("content"),
                        "source": meta.get("source"),
                    }
                )

        return retrieved_texts


def main():
    pass

    # em = EmbeddingManager()
    # vs = FaissVectorStore(vector_dimension=384)

    # question = ["Nimesh"]
    # embedded_question = em.generate_embedding(question)
    # result = vs.retreive_documents(embedded_question)
    # print(result)


if __name__ == "__main__":
    main()
