from pathlib import Path
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

""" 
def load_documents_directory_loader(data_dir: Path):
    "Using Directory loader instead of Individual loader"
    documents = []
    dir_loader = DirectoryLoader(
        path=data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents.extend(dir_loader.load())
    return documents

"""


def update_metadata(loader, path):
    for docs in loader:
        docs.metadata.update(
            {
                "source": str(path),
                "file_name": path.name,
                "extension": path.suffix,
            }
        )


def load_file(loader_type, files_path, encoding=None):
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


def load_pdf_file(data_dir):
    files_path = list(data_dir.glob("**/*.pdf"))
    print(f"{len(files_path)} pdf/(s) to process.. ")
    documents = load_file(PyPDFLoader, files_path)
    print(
        f"Total {len(documents)} documents processed ✓ from {len(files_path)} pdf files.."
    )
    return documents


def load_text_file(data_dir):
    files_path = list(data_dir.glob("**/*.txt"))
    print(f"{len(files_path)} txt/(s) to process.. ")
    documents = load_file(TextLoader, files_path, encoding="utf-8")
    print(
        f"Total {len(documents)} documents processed ✓ from {len(files_path)} txt files.."
    )
    return documents


def load_all_documents(data_dir, file_types):
    print(f"\n{"*"*50}\nDATA INGESTION PIPELINE \n{"*"*50}\n")
    loaders = {"txt": load_text_file, "pdf": load_pdf_file}
    data_path = Path(data_dir)
    if not data_path.exists():
        print("Nothing found inside path")
        return None
    all_documents = []
    for type in file_types:
        loader_func = loaders[type]
        document = loader_func(data_path)
        all_documents.extend(document)
    print(
        f"\nFinal: loaded {len(all_documents)} source document page(s) from file types: {', '.join(file_types)} ✓ \n "
    )

    return all_documents


def main():
    data_dir = Path("../data")
    load_all_documents(data_dir, file_types=["txt", "pdf"])


if __name__ == "__main__":
    main()

