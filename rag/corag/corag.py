import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..splitter import split_chunks_from_pages
from ..loader import extract_content
from ..prompt import combine_prompt
from .decompose import decompose
from .search import retrieve_per_subquestion
from .subanswer import answer_subquestions
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


def ask_pdf_corag(
    file_path: str,
    question: str,
    progress=None,
    max_hops: int = 3,
    k: int = 3,
) -> str:
    # ── Phase 1: Indexing ─────────────────────────────────────────────────────

    db = get_or_create_vector_db(file_path, embeddings_model, progress=progress)

    # ── Phase 2: Devide sub questions ──────────────────────────────────────
    if progress:
        progress(60, "Đang chia thành các câu hỏi con...")

    sub_questions = decompose(question, llm)
    print("Sub questions:", sub_questions)
    if progress:
        progress(85, "AI đang trả lời các câu hỏi con...")
    
    retrieved = retrieve_per_subquestion(
        sub_questions=sub_questions,
        vectorstore=db,
        k=3
    )
    answers = answer_subquestions(retrieved, llm)

    # ── Phase 3: Generation ───────────────────────────────────────────────────
    if progress:
        progress(95, "AI đang tổng hợp câu trả lời...")

    prompt = combine_prompt( question,answers)

    response = llm.invoke(prompt).content

    if progress:
        progress(100, "Hoàn thành")

    return response