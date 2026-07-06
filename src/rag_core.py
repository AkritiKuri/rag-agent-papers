"""
rag_core.py
Core RAG logic: load a FAISS index, retrieve top-k chunks for a query,
and ask the LLM to answer using ONLY those chunks, citing the source
paper for each claim.
"""
import os
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

ANSWER_SYSTEM_PROMPT = """You are a research assistant answering questions \
about a small corpus of NLP/ML papers. You will be given retrieved excerpts \
from those papers, each tagged with its source paper name.

Rules:
- Answer ONLY using the provided excerpts. If the excerpts don't contain \
the answer, say so plainly instead of guessing.
- You may ONLY cite paper names that appear in the ALLOWED CITATIONS list \
below. NEVER invent, guess, or reuse a paper name from outside this list, \
even if it seems familiar to you.
- After every claim, cite the source paper in square brackets, e.g. \
[paper_name].
- Be concise and technical; assume the reader is an ML practitioner.
"""


@dataclass
class RetrievedChunk:
    text: str
    source_paper: str
    score: float


class RAGPipeline:
    def __init__(
        self,
        index_path: str = "data/faiss_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "phi3:mini",
        retrieval_k: int = 4,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = FAISS.load_local(
            index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retrieval_k = retrieval_k
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = ChatOllama(model=llm_model, temperature=0, base_url=ollama_base_url)

    def retrieve(self, query: str, k: int = None) -> List[RetrievedChunk]:
        k = k or self.retrieval_k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(
                text=doc.page_content,
                source_paper=doc.metadata.get("source_paper", "unknown"),
                score=score,
            )
            for doc, score in results
        ]

    def answer(self, query: str, k: int = None) -> dict:
        chunks = self.retrieve(query, k=k)
        context = "\n\n".join(
            f"[{c.source_paper}]: {c.text}" for c in chunks
        )
        allowed = sorted({c.source_paper for c in chunks})
        allowed_line = "ALLOWED CITATIONS (only these): " + ", ".join(allowed)
        messages = [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", f"{allowed_line}\n\nExcerpts:\n\n{context}\n\nQuestion: {query}"),
        ]
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": allowed,
            "chunks_used": len(chunks),
        }


def load_pipeline(**kwargs) -> RAGPipeline:
    return RAGPipeline(**kwargs)