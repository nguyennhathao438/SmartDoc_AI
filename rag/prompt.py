
from langdetect import detect
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
from langdetect import detect

def build_prompt_graphrag(context, question):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là AI trả lời câu hỏi dựa trên dữ liệu truy xuất từ hệ thống GraphRAG.

Hướng dẫn:
- Ngữ cảnh gồm 2 loại:
  + [TEXT]&#58; thông tin mô tả
  + [GRAPH]&#58; quan hệ giữa các thực thể (dạng A --(RELATION)--> B)
- Hãy CHUYỂN các quan hệ GRAPH thành câu tự nhiên (KHÔNG giữ ký hiệu [], --, hoặc RELATION).
- Ưu tiên sử dụng thông tin từ [GRAPH] để suy luận (vì nó thể hiện quan hệ rõ ràng)
- Kết hợp với [TEXT] để bổ sung ý nghĩa
- Trả lời ngắn gọn 3-4 câu bằng tiếng Việt.

Nếu không tìm thấy thông tin thì trả lời:
"Tôi không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
    else:
        return f"""
You are an AI assistant using GraphRAG retrieved data.

Instructions:
- Context includes:
  + [TEXT]&#58; descriptive content
  + [GRAPH]&#58; relationships (A --(RELATION)--> B)
- Convert GRAPH relationships into natural language (DO NOT include [], --, or RELATION).
- Prioritize reasoning from GRAPH relationships
- Use TEXT to support explanation
- Answer in 3-4 sentences clearly

If not found:
"I could not find the information in the document."

Context:
{context}

Question:
{question}
"""