from ..splitter import split_chunks_from_pages
from ..loader import extract_pages_from_pdf
from ..prompt import build_prompt_graphrag
import streamlit as st
from .extractgraph import extract_graph_documents
from .vectorstore import build_faiss_node_index

from .mapping import map_nodes_to_source_texts
from .networkx import build_networkx_graph
from .search import search_graph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=api_key,model="llama-3.3-70b-versatile", temperature=0)
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Concept", "Process", "Method", "Example", "Component"],
    allowed_relationships=["INCLUDES", "DESCRIBES", "IS_EXAMPLE_OF", "APPLIES_TO"]
)
def ask_pdf(file_path, question, progress=None):
    if progress:
        progress(10,"Đang đọc PDF ...")
    pages = extract_pages_from_pdf(file_path)
    if progress:
        progress(30, "Đang chia nhỏ tài liệu...")
    
    chunks = split_chunks_from_pages(pages,
    chunk_size=st.session_state.chunk_size,
    chunk_overlap=st.session_state.chunk_overlap)
    if progress:
        progress(60, "Đang trích xuất tri thức...")
    graph_docs = extract_graph_documents(chunks, transformer)
    node_mapping = map_nodes_to_source_texts(graph_docs)
    networkx_graph = build_networkx_graph(graph_docs)
    node_vector_store = build_faiss_node_index(graph_docs, embeddings_model)
    if progress:
        progress(80, "Đang tìm kiếm thông tin liên quan...")
    context = search_graph(question, networkx_graph, node_mapping, node_vector_store, hop=1, max_context=20)
    if progress:
        progress(90, "AI đang tạo câu trả lời...")
    prompt = build_prompt_graphrag(context, question)
    print("Prompt:", prompt)
    # llm_answer = OllamaLLM(
    # model="qwen2.5:7b",
    # temperature=st.session_state.temperature,
    # top_p=st.session_state.top_p,
    # repeat_penalty=st.session_state.repeat_penalty
    # )
    # print("param LLM:", st.session_state.temperature, st.session_state.top_p, st.session_state.repeat_penalty)
    response = llm.invoke(prompt).content
    if progress:
        progress(100, "Hoàn thành")

    return response
