# rag_2

Simple Retrieval-Augmented Generation (RAG) project with two vector store backends:
- ChromaDB
- FAISS

It loads local PDF/TXT files, chunks them, embeds text with `all-MiniLM-L6-v2`, retrieves relevant chunks, and asks Groq for final answers.

## Current Project Structure

```text
rag_2/
|-- data/
|   |-- Anschreiben_für_Bewerbung_Nimesh.pdf
|   `-- text_one.txt
|-- notebook/
|   |-- document.ipynb
|   `-- pdf_loader.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- embeddings.py
|   |-- vectorstore_chromadb.py
|   |-- vectorstore_faiss.py
|   `-- response_models/
|       |-- groq_response.py
|       `-- prompt_utils.py
|-- store/
|   `-- vector_store/
|       |-- chromadb/
|       `-- faiss/
|-- .env
|-- .gitignore
|-- main.py
`-- README.md
```

## What Each Module Does

- `src/data_loader.py`
  - Recursively loads `*.pdf` and `*.txt` files from `data/`.
  - Adds metadata keys: `source`, `file_name`, `extension`.

- `src/embeddings.py`
  - Uses `SentenceTransformer("all-MiniLM-L6-v2")`.
  - Splits documents with `RecursiveCharacterTextSplitter` (`chunk_size=2000`, `chunk_overlap=200`).
  - Returns `(chunks, embeddings)`.

- `src/vectorstore_chromadb.py`
  - Persists vectors in `store/vector_store/chromadb/`.
  - Uses deterministic SHA256 IDs for chunks.
  - Retrieves documents with distance and similarity score.

- `src/vectorstore_faiss.py`
  - Persists FAISS index + metadata in `store/vector_store/faiss/`.
  - Normalizes embeddings with L2.
  - Stores metadata in `faiss_metadata.json`.

- `src/response_models/prompt_utils.py`
  - Builds retrieval context.
  - Builds prompt for grounded answer generation.

- `src/response_models/groq_response.py`
  - Wraps `ChatGroq` (`llama-3.1-8b-instant` by default).

- `main.py`
  - Runs both ingestion pipelines (Chroma and FAISS).
  - Runs both QA flows (Chroma and FAISS).

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install langchain-core langchain-community langchain-groq sentence-transformers chromadb python-dotenv langchain-text-splitters pypdf numpy faiss-cpu
```

3. Create `.env` in project root.

```env
API_KEY=your_groq_api_key_here
```

## Run

Run from project root:

```powershell
python main.py
```

Current `main.py` flow is:
1. Ingest to ChromaDB.
2. Ingest to FAISS.
3. Ask question with Chroma retrieval.
4. Ask question with FAISS retrieval.

## Storage Paths

- Chroma artifacts: `store/vector_store/chromadb/`
- FAISS artifacts:
  - `store/vector_store/faiss/faiss_index.bin`
  - `store/vector_store/faiss/faiss_metadata.json`

## Notes and Caveats

- Run from the repository root so `from src...` imports resolve correctly.
- FAISS query embedding should be 2D (`[query]`), not raw string, to avoid `normalize_L2` shape errors.
- `main.py` currently re-ingests on each run.
  - Chroma uses stable IDs, so upsert behavior is safer.
  - FAISS appends vectors each run and can accumulate duplicates.
- `src/vectorstore_faiss.py` method name is `retreive_documents` (typo in method name, but used consistently).
- If `API_KEY` is missing, Groq initialization will fail.

## Notebooks

- `notebook/document.ipynb`: basic `Document` structure experiments.
- `notebook/pdf_loader.ipynb`: early pipeline scratch notebook.

## Suggested Next Improvements

1. Add `requirements.txt` or `pyproject.toml`.
2. Add CLI flags to choose backend and mode (`ingest` vs `query`).
3. Add dedup/checkpoint logic for FAISS ingestion.
4. Add tests for loader, embedding, and retrieval logic.
