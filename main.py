import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingManager
from src.vectorestore_chromadb import VectorStore
from src.response_models.groq_response import GenerateGroqResponse
from src.response_models.prompt_utils import create_prompt, create_context

load_dotenv()


def data_saver():
    data_dir = Path("data")
    documents = load_all_documents(data_dir, file_types=["txt", "pdf"])
    em = EmbeddingManager()
    embedded_documents = em.embed_documents(documents=documents)
    store = VectorStore()
    breakpoint()
    store.add_documents(embedded_documents[0], embedded_documents[1])


def get_answer_chroma(query, api_key=""):
    em = EmbeddingManager()
    embedded_query = em.generate_embedding(query)
    store = VectorStore()
    retrieved_documents = store.retrieve_documents(embedded_query, top_k=1)
    context = create_context(retrieved_docs=retrieved_documents, detail_mode=False)
    prompt = create_prompt(question=query, context=context)
    llm = GenerateGroqResponse(api_key=api_key)
    answer = llm.generate_answer(prompt=prompt)
    print(answer)


def main():
    # data_saver()
    api_key = os.environ.get("API_KEY")
    get_answer_chroma(
        query="Who is Nimesh Ghimire ? ",
        api_key=api_key,
    )

    # response = ""
    # if response:
    #     print("Answer:", response["answer"])
    #     print("Sources:", response["sources"])
    #     print("Confidence:", response["confidence"])
    #     print("Context Preview:", response["context"][:300])
    # else:
    #     print("Nothing found")


if __name__ == "__main__":
    main()
