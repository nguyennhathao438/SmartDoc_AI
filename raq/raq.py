from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import streamlit as st
from langdetect import detect, DetectorFactory

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def split_chunks_from_pages(pages_text, chunk_size=500, chunk_overlap=50):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print("Chunk size:", chunk_size)
    chunks = []

    for page in pages_text:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return chunks

def extract_pages_from_pdf(file_path):
    pages_text = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append({
                    "page": i + 1,
                    "text": text
                })

    return pages_text
def build_prompt(context, question):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên nội dung ngữ cảnh.
Trả lời ngắn gọn 3-4 câu bằng tiếng Việt.
Nếu không tìm thấy thông tin trong ngữ cảnh thì hãy nói:
"Tôi không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
    else:
        return f"""
You are an AI assistant that helps answer questions based on the provided context.
Answer briefly in 3-4 sentences in English.
If the information is not found in the context, please say:
"I could not find the information in the document."

Context:
{context}

Question:
{question}
"""


def ask_pdf(file_path, question, progress=None):

    if progress:
        progress(10, "Đang đọc PDF...")

    pages = extract_pages_from_pdf(file_path)

    if progress:
        progress(30, "Đang chia nhỏ tài liệu...")

    chunks = split_chunks_from_pages(pages,
    chunk_size=st.session_state.chunk_size,
    chunk_overlap=st.session_state.chunk_overlap)

    texts = [chunk["text"] for chunk in chunks]

    if progress:
        progress(60, "Đang tạo embeddings...")

    db = FAISS.from_texts(texts, embeddings_model)

    if progress:
        progress(75, "Đang tìm nội dung liên quan...")

    docs = db.similarity_search(question, k=3)

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