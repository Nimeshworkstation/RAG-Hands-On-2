from pathlib import Path
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def load_documents(data_dir: Path) -> List[Document]:
    "Using individual loader"
    documents: List[Document] = []
    len_total_files = len([item for item in data_dir.iterdir()])
    print(f"found {len_total_files} files/objects to process")

    for file_path in data_dir.iterdir():

        if not file_path.is_file():
            print(
                f"{file_path} : is not a file  ! Processing Aborting ..",
            )
            continue
        file_ext = ((str(file_path).split("."))[-1]).lower()
        if file_ext not in ["txt", "pdf"]:
            print(f"Wrong file format {file_path} ! Processing Aborting ..")
            continue
        print(f"processing {file_path.name} ")
        try:
            if file_ext == "pdf":
                doctype = PyPDFLoader(file_path=file_path)
                loader = doctype.load()
            elif file_ext == "txt":
                doctype = TextLoader(file_path=file_path, encoding="utf-8")
                loader = doctype.load()
            else:
                continue
        except Exception as e:
            print("Error occured while loading document from !!", file_path, e)
            continue
        for docs in loader:
            docs.metadata.update(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "extension": file_ext,
                }
            )
        documents.extend(loader)
        print(f" ✓ loaded {len(loader)} pages..")
    print(f"Total {len(documents)} documents processed ✓")
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


def main():
    data_dir = Path("data")

    if data_dir.exists():
        data = load_documents(data_dir)

    chunks = split_chunk(data)
    for item in chunks:
        print(item)
        print("\n")
    print(len(chunks))


if __name__ == "__main__":
    main()
