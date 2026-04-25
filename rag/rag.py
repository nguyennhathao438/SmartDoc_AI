from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import streamlit as st
from .splitter import split_chunks_from_pages
from .loader import extract_content
from .prompt import build_prompt
from langchain_core.documents import Document

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def ask_pdf(file_path, question, progress=None):

    if progress:
        progress(10, "Đang đọc PDF...")

    pages = extract_content(file_path)

    if progress:
        progress(30, "Đang chia nhỏ tài liệu...")

    chunks = split_chunks_from_pages(pages,
    chunk_size=st.session_state.chunk_size,
    chunk_overlap=st.session_state.chunk_overlap)

    docs = [
    Document(
        page_content=chunk["text"],
        metadata={"source": chunk["source"]}
    )
    for chunk in chunks
]

    if progress:
        progress(60, "Đang tạo embeddings...")

    db = FAISS.from_documents(docs, embeddings_model)

    if progress:
        progress(75, "Đang tìm nội dung liên quan...")

    docs = db.similarity_search(question, k=3)
    print("Docs:", docs)
    context = "\n\n".join([doc.page_content for doc in docs])

    if progress:
        progress(90, "AI đang tạo câu trả lời...")

    prompt = build_prompt(context, question)

    llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=st.session_state.temperature,
    top_p=st.session_state.top_p,
    repeat_penalty=st.session_state.repeat_penalty
    )
    print("param LLM:", st.session_state.temperature, st.session_state.top_p, st.session_state.repeat_penalty)
    response = llm.invoke(prompt)

    if progress:
        progress(100, "Hoàn thành")

    return response