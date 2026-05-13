from pathlib import Path
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def update_metadata(loader, path):
    for docs in loader:
        docs.metadata.update(
            {
                "source": str(path),
                "file_name": path.name,
                "extension": path.suffix,
            }
        )


def list_files(loader_type, files_path, encoding=None):
    documents = []
    for file_path in files_path:
        try:
            kwargs = {"file_path": str(file_path)}
            if encoding:
                kwargs["encoding"] = encoding
            doctype = loader_type(**kwargs)
        except Exception as e:
            print(f" ✗ failed to load {file_path.name}: {e}")
            continue
        loader = doctype.load()
        print(f" ✓ loaded {len(loader)} pages from {file_path.name}..")
        update_metadata(loader, file_path)
        documents.extend(loader)
    return documents


def list_pdf_files(data_dir):
    files_path = list(data_dir.glob("**/*.pdf"))
    print(f"{len(files_path)} pdf/(s) to process.. ")
    documents = list_files(PyPDFLoader, files_path)
    print(
        f"Total {len(documents)} documents processed ✓ from {len(files_path)} pdf files.."
    )
    return documents


def list_txt_files(data_dir):
    files_path = list(data_dir.glob("**/*.txt"))
    print(f"{len(files_path)} txt/(s) to process.. ")
    documents = list_files(TextLoader, files_path, encoding="utf-8")
    print(
        f"Total {len(documents)} documents processed ✓ from {len(files_path)} txt files.."
    )
    return documents


def split_chunk(
    document: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200
):
    total_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    chunks = text_splitter.split_documents(documents=document)
    total_chunks.extend(chunks)
    print(f"Splitted {len(document)} Documents into {len(total_chunks)} chunks ✓ \n")
    return total_chunks


def load_all_documents(data_dir, *args):
    all_documents = []
    for list_file in args:
        document = list_file(data_dir)
        all_documents.extend(document)
    return all_documents


def main():
    data_dir = Path("../data")

    if data_dir.exists():
        result = load_all_documents(data_dir, list_pdf_files, list_txt_files)
        chunks = split_chunk(result)
        for item in chunks:
            print(item)
            print("\n")
        print(len(chunks))


if __name__ == "__main__":
    main()
