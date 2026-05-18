def create_context(self, retrieved_docs):
    if not retrieved_docs:
        print("⚠ No documents retrieved for context")
        return
    context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])
    return context


def create_prompt(self, question: str, context: str):
    if not context:
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
        {context}

        Question:
        {question}

        Answer:
        

        """
