from multiprocessing import Process, Queue
from rag.rag import ask_pdf
from rag.graph_rag.graphrag import ask_pdf as ask_pdf_graph
from rag.corag.corag import ask_pdf_corag
def run_ai_pipeline(file_path, question, mode, progress=None):
    rag_response = None
    graph_response = None
    corag_response = None

    def run_rag():
        nonlocal rag_response
        rag_response = ask_pdf(file_path, question,progress=progress)

    def run_graph():
        nonlocal graph_response
        graph_response = ask_pdf_graph(file_path, question, progress=progress)

    def run_corag():
        nonlocal corag_response
        corag_response = ask_pdf_corag(file_path, question, progress=progress)

    if mode == "RAG":
        return ask_pdf(file_path, question,progress=progress)

    elif mode == "GraphRAG":
        return ask_pdf_graph(file_path, question, progress=progress)
        
    elif mode == "CoRAG":
        return ask_pdf_corag(file_path, question, progress=progress)
