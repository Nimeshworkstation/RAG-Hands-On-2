from vectorstore import VectorStore
from embedder import EmbeddingManager


class RAGRetrival:
    """Handles query-based retrival from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.collection = self.vector_store.client.get_collection(
            self.vector_store.collection_name
        )

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        retrieved_docs = []
        query_embedding = self.embedding_manager.generate_embedding([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if result.get("documents") and result.get("documents")[0]:
            documents = result.get("documents")[0]
            metadatas = result.get("metadatas")[0]
            distances = result.get("distances")[0]
            ids = result.get("ids")[0]
            print(distances, "===========================")

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
                            "rank": index + 1,
                        }
                    )
            print(f"Retrieved {len(retrieved_docs)} docuemnt (after filtering)")
        else:
            print("No document found")

        return retrieved_docs


def main():
    vector_store = VectorStore()
    embedding_manager = EmbeddingManager()
    embedded_query = RAGRetrival(vector_store, embedding_manager)
    data = embedded_query.retrieve("Initiativbewerbung als Software-Entwickler")
    print(data)


if __name__ == "__main__":
    main()
