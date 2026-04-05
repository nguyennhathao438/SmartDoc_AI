import pdfplumber
from .splitter import split_chunks_from_pages
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
