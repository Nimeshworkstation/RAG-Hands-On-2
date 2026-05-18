from langchain_groq import ChatGroq


class GenerateGroqResponse:
    """Handles LLM initialization and response generation using Groq."""

    def __init__(self, api_key="", model="llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.context = ""
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

    def create_context(self, retrieved_docs):
        if not retrieved_docs:
            print("⚠ No documents retrieved for context")
            return
        self.context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])

    def create_prompt(self, question: str):
        if not self.context:
            print("⚠ Not enough context found to answer the question")
            return None
        if not question.strip():
            print("")
            return

        return f"""
            You are a helpful assistant. Answer ONLY from the provided context.
            If the context is not enough, say exactly: "I don't have enough context."
            Keep the answer concise and factual.

            Context:
            {self.context}

            Question:
            {question}

            Answer:
            

            """

    def generate_answer(self, prompt):
        print("💭 Generating answer...\n")
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
        return response.content


def main():
    pass


if __name__ == "__main__":
    main()
