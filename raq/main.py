from loader import extract_pages_from_pdf
from splitter import split_chunks_from_pages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def get_embeddings(file_path):
    pages = extract_pages_from_pdf(file_path)
    chunks = split_chunks_from_pages(pages)
    texts = [chunk["text"] for chunk in chunks]
    return texts

def save_vector_store(texts):
    db = FAISS.from_texts(texts, embeddings_model)
    db.save_local("vector_store")
    print("Vector store created!")

def load_vector_store():
    db = FAISS.load_local(
        "vector_store",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    return db
def build_prompt(context, question):
    prompt = f"""
Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên tài liệu.

Tài liệu:
{context}

Câu hỏi:
{question}

Hãy trả lời dựa trên nội dung tài liệu.
Nếu không tìm thấy thông tin trong tài liệu thì hãy nói:
"Tôi không tìm thấy thông tin trong tài liệu."
"""
    return prompt
if __name__ == "__main__":
    embeddings = get_embeddings("./data/tkgt_doc.pdf")
    save_vector_store(embeddings)
    db = load_vector_store()

    docs = db.similarity_search("Mô hình xử lý thông tin của con người", k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt(context, "Mô hình xử lý thông tin của con người")
    llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
    )
    response = llm.invoke(prompt)

    print(response)