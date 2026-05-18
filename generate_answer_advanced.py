import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vectorstore import VectorStore
from embedder import EmbeddingManager
from rag_retrieval import RAGRetreival

load_dotenv()


class GenerateGroqResponse:
    """Handles LLM initialization and response generation using Groq."""

    def __init__(self, api_key="", model="", return_context=False):
        self.api_key = api_key
        self.model = model
        self.context = ""
        self.confidence = 0.0
        self.sources = []
        self.return_context = return_context
        self._initialize_llm()

    def _initialize_llm(self):
        if not self.api_key:
            raise ValueError("API key not provided")
        if not self.model:
            raise ValueError("model not provided")
        try:
            self.llm = ChatGroq(
                api_key=self.api_key, model=self.model, temperature=0.1, max_tokens=1024
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

        print(f"✓ Initialized Groq LLM with model: {self.model}")

    def create_detail_context(self, retrieved_docs):
        if not retrieved_docs:
            print("⚠ No documents retrieved for context")
            return self.context
        self.context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])
        self.sources = [
            {
                "source": doc.get("metadata").get("source"),
                "score": doc.get("similiarity_score", ""),
                "page": doc.get("metadata").get("page"),
                "preview": doc.get("content", "") + " ...",
            }
            for doc in retrieved_docs
        ]
        self.confidence = max([doc["similiarity_score"] for doc in retrieved_docs])

    def generate_detail_answer(self, prompt):
        if not prompt:
            print("⚠ No Prompts generated for the question..")
            return None
        print("💭 Generating answer...\n")
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
        output = {
            "answer": response.content,
            "sources": self.sources,
            "confidence": self.confidence,
            "context": self.context if self.return_context else "",
        }

        return output


def main():
    api_key = os.environ.get("API_KEY", "")

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = RAGRetreival(vector_store, embedding_manager)
    query = "Who is Deepsikha Kafle?"
    retrieved_docs = retriever.retrieve(query=query, top_k=3)
    groq_llm = GenerateGroqResponse(
        api_key=api_key, model="llama-3.1-8b-instant", return_context=True
    )
    groq_llm.create_context(retrieved_docs)
    prompt = groq_llm.create_prompt(question=query)
    response = groq_llm.generate_answer(prompt)
    breakpoint()
    if response:
        print("Answer:", response["answer"])
        print("Sources:", response["sources"])
        print("Confidence:", response["confidence"])
        print("Context Preview:", response["context"][:300])
    else:
        print("Nothing found")


if __name__ == "__main__":
    main()
