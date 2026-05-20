import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingManager
from src.vectorstore_chromadb import ChromaVectorStore
from src.vectorstore_faiss import FaissVectorStore
from src.response_models.groq_response import GenerateGroqResponse
from src.response_models.gemini_response import GenerateGeminiResponse
from src.response_models.prompt_utils import create_prompt, create_context

load_dotenv()


def data_saver_chromadb():
    data_dir = Path("data")
    documents = load_all_documents(data_dir, file_types=["txt", "pdf"])
    em = EmbeddingManager()
    embedded_documents = em.embed_documents(documents=documents)
    store = ChromaVectorStore()
    store.add_documents(embedded_documents[0], embedded_documents[1])


def data_saver_faiss():
    data_dir = Path("data")
    documents = load_all_documents(data_dir, file_types=["txt", "pdf"])
    em = EmbeddingManager()
    embedded_documents = em.embed_documents(documents=documents)
    store = FaissVectorStore()
    store.save_index_metadata(embedded_documents)


def get_answer_groq_chromadb(query, api_key=""):
    em = EmbeddingManager()
    embedded_query = em.generate_embedding(query)
    store = ChromaVectorStore()
    retrieved_documents = store.retrieve_documents(embedded_query, top_k=1)
    context = create_context(retrieved_docs=retrieved_documents, detail_mode=False)
    prompt = create_prompt(question=query, context=context)
    llm = GenerateGroqResponse(api_key=api_key)
    answer = llm.generate_answer(prompt=prompt)
    print(answer)


def get_answer_groq_faiss(query, api_key=""):
    em = EmbeddingManager()
    embedded_query = em.generate_embedding([query])
    store = FaissVectorStore()
    retrieved_documents = store.retreive_documents(embedded_query)
    context = create_context(retrieved_docs=retrieved_documents, detail_mode=False)
    prompt = create_prompt(question=query, context=context)
    llm = GenerateGroqResponse(api_key=api_key)
    answer = llm.generate_answer(prompt=prompt)
    print(answer)


def get_answer_gemini_faiss(query, api_key=""):
    em = EmbeddingManager()
    embedded_query = em.generate_embedding([query])
    store = FaissVectorStore()
    retrieved_documents = store.retreive_documents(embedded_query)
    context = create_context(retrieved_docs=retrieved_documents, detail_mode=False)
    prompt = create_prompt(question=query, context=context)
    llm = GenerateGeminiResponse(api_key=api_key)
    answer = llm.generate_answer(prompt=prompt)
    print(answer)


def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    # data_saver_chromadb()
    # data_saver_faiss()
    # get_answer_groq_chromadb(
    #     query="Who is Nimesh Ghimire ? ",
    #     api_key=api_key,
    # )

    get_answer_gemini_faiss(query="Who is Nimesh Ghimire ?", api_key=gemini_api_key)


if __name__ == "__main__":
    main()
