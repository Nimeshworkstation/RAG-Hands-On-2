# rag_2

A simple Retrieval-Augmented Generation (RAG) pipeline using:
- LangChain document loaders
- SentenceTransformers for embeddings
- ChromaDB as vector store
- Groq (`langchain_groq`) for answer generation

## Project Structure

- `main.py`: entry point (ingestion + query flow)
- `src/data_loader.py`: loads `.pdf` and `.txt`, enriches metadata
- `src/embeddings.py`: chunks documents and generates embeddings
- `src/vectorestore_chromadb.py`: stores/retrieves vectors with ChromaDB
- `src/response_models/prompt_utils.py`: shared prompt/context builders
- `src/response_models/groq_response.py`: Groq LLM wrapper
- `data/`: local source documents

## How It Works

1. Load documents from `data/` (`.txt` and `.pdf`).
2. Split documents into chunks.
3. Generate embeddings for chunks.
4. Upsert chunks + embeddings into ChromaDB.
5. For a query:
   - Embed the query
   - Retrieve top-k relevant chunks
   - Build context + prompt
   - Ask Groq model and print answer

## Setup

## 1) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install langchain-core langchain-community langchain-groq sentence-transformers chromadb python-dotenv langchain-text-splitters pypdf numpy
```

## 3) Configure environment variables

Create `.env` in project root:

```env
API_KEY=your_groq_api_key_here
```

## Run

`main.py` currently calls query flow by default.

```powershell
python main.py
```

## Ingest Documents Into ChromaDB

In `main.py`, uncomment `data_saver()` inside `main()` to ingest documents first:

```python
def main():
    data_saver()
    api_key = os.environ.get("API_KEY")
    get_answer_chroma(query="Who is Nimesh Ghimire ? ", api_key=api_key)
```

Then run:

```powershell
python main.py
```

## Notes / Current Caveats

- In `src/vectorestore_chromadb.py`, default `persist_directory` is `"..data/vector_store/chromadb/"`. Consider changing to `"data/vector_store/chromadb/"`.
- `prompt_utils.create_context()` expects key `similiarity_score`, while retrieval output uses `similarity_score`. This key mismatch can break detail mode.
- `main.py` includes a `breakpoint()` in `data_saver()`. Remove it for normal runs.
- If no documents are ingested yet, retrieval may return empty results.

## Quick API Usage (Programmatic)

```python
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingManager
from src.vectorestore_chromadb import VectorStore
from src.response_models.prompt_utils import create_context, create_prompt
from src.response_models.groq_response import GenerateGroqResponse

# ingest
docs = load_all_documents("data", ["txt", "pdf"])
em = EmbeddingManager()
chunks, vectors = em.embed_documents(docs)
store = VectorStore()
store.add_documents(chunks, vectors)

# ask
query = "Who is Nimesh Ghimire?"
q_vec = em.generate_embedding(query)
hits = store.retrieve_documents(q_vec, top_k=1)
context = create_context(hits)
prompt = create_prompt(question=query, context=context)
llm = GenerateGroqResponse(api_key="<API_KEY>")
answer = llm.generate_answer(prompt)
print(answer)
```

## Future Improvements

- Add `requirements.txt` or `pyproject.toml`
- Add CLI flags for ingest vs query mode
- Add better logging instead of `print`
- Add unit tests for loaders, embedding pipeline, and prompt utilities
- Add support for additional LLM providers
