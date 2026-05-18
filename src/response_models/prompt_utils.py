def create_prompt(question: str, context: str):
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


def create_context(retrieved_docs, detail_mode=False):
    details = {}
    if not retrieved_docs:
        print("⚠ No documents retrieved for context")
        return ""
    context = "\n\n".join([doc.get("content", "") for doc in retrieved_docs])
    if not detail_mode:
        return context
    details["sources"] = [
        {
            "source": doc.get("metadata").get("source"),
            "score": doc.get("similiarity_score", ""),
            "page": doc.get("metadata").get("page"),
            "preview": doc.get("content", "") + " ...",
        }
        for doc in retrieved_docs
    ]
    details["confidence"] = max([doc["similiarity_score"] for doc in retrieved_docs])
    details["context"] = context
    return details
