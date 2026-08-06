import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.rag.retriever_setup import get_vector_store


def load_document(file_path: str) -> List[Document]:
    """
    Loads text from a PDF or TXT file into a list of LangChain Document objects.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .pdf and .txt are supported.")

    return loader.load()


def process_and_upload_document(file_path: str, description: str = "") -> int:
    """
    Loads, chunks, embeds, and indexes a document into Qdrant Vector DB.

    Args:
        file_path: Path to the PDF or TXT file.
        description: Brief user description of the document.

    Returns:
        int: Number of text chunks indexed into Qdrant.
    """
    # 1. Load document text
    raw_documents = load_document(file_path)

    # 2. Configure Recursive Text Splitter (1000 char chunks with 200 overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # 3. Split raw documents into contextual chunks
    chunks = text_splitter.split_documents(raw_documents)

    # 4. Attach metadata (description and filename) to each chunk
    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["description"] = description
        chunk.metadata["source"] = filename

    # 5. Add document chunks to Qdrant Vector Store
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)