import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..splitter import split_chunks_from_pages
from ..loader import extract_content
from ..prompt import build_prompt

from .searchloop import run_corag_loop
from ..rag import get_or_create_vector_db
load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


def _build_faiss(chunks: list[dict]) -> FAISS:
    docs = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"]},
        )
        for c in chunks
    ]
    return FAISS.from_documents(docs, embeddings_model)


def ask_pdf_selfrag(
    file_path: str,
    question: str,
    progress=None,
    max_hops: int = 3,
    k: int = 3,
) -> str:
    # ── Phase 1: Indexing ─────────────────────────────────────────────────────

    db = get_or_create_vector_db(file_path, embeddings_model, progress=progress)

    # ── Phase 2: Chain-of-Retrieval loop ──────────────────────────────────────
    context_buffer = run_corag_loop(
        db=db,
        llm=llm,
        question=question,
        max_hops=max_hops,
        k=k,
        progress=progress,
    )

    # ── Phase 3: Generation ───────────────────────────────────────────────────
    if progress:
        progress(85, "AI đang tạo câu trả lời...")

    final_context = "\n\n".join(context_buffer)
    prompt = build_prompt(final_context, question)

    print(f"[CoRAG] Tổng context: {len(context_buffer)} đoạn | {len(prompt)} chars")

    response = llm.invoke(prompt).content

    if progress:
        progress(100, "Hoàn thành")

    return response