import pdfplumber
import docx
import os

def extract_content(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        return extract_pages_from_pdf(file_path)
    elif ext == ".docx":
        return extract_paragraphs_from_docx(file_path)
    else:
        print(f"Định dạng {ext} chưa được hỗ trợ.")
        return []

def extract_pages_from_pdf(file_path):
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append({
                    "source": f"Page {i + 1}",
                    "text": text.strip()
                })
    return pages_text

def extract_paragraphs_from_docx(file_path):
    paragraphs_text = []
    doc = docx.Document(file_path)
    
    for i, para in enumerate(doc.paragraphs):
        if para.text and para.text.strip():
            paragraphs_text.append({
                "source": f"Paragraph {i + 1}",
                "text": para.text.strip()
            })
    return paragraphs_text