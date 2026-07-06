"""
ingestion.py
Loads papers (PDF or TXT) from data/papers/, chunks them, embeds with a
local sentence-transformers model, and builds/saves a FAISS index.

Usage:
    python -m src.ingestion --chunk-size 1000 --chunk-overlap 200 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --index-path data/faiss_index
"""
import argparse
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"


def load_documents(papers_dir: Path):
    """Load every .pdf and .txt file in papers_dir, tagging each chunk
    with its source filename so we can cite it later."""
    docs = []
    for path in sorted(papers_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_paper"] = path.stem
        docs.extend(loaded)
    if not docs:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {papers_dir}. "
            "Drop your research papers there before ingesting."
        )
    return docs


def chunk_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(chunks, embedding_model: str, index_path: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore


def run_ingestion(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "data/faiss_index",
    papers_dir: Path = PAPERS_DIR,
):
    docs = load_documents(papers_dir)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    build_index(chunks, embedding_model, index_path)
    return {
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--embedding-model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--index-path", type=str, default="data/faiss_index")
    args = parser.parse_args()

    stats = run_ingestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        index_path=args.index_path,
    )
    print(f"Ingested {stats['num_documents']} docs -> {stats['num_chunks']} chunks")
    print(f"Index saved to {args.index_path}")
