
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
# CORAG
def sufficiency_prompt(question, context):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là một trợ lý đánh giá chất lượng ngữ cảnh.
 
Câu hỏi gốc: {question}
 
Ngữ cảnh hiện tại đã thu thập:
{context}
 
Hãy đánh giá xem ngữ cảnh trên có đủ để trả lời câu hỏi không.
Trả lời ĐÚNG theo định dạng JSON sau, KHÔNG thêm gì khác:
{{
  "sufficient": true hoặc false,
  "reason": "lý do ngắn gọn",
  "refined_query": "câu truy vấn mới nếu cần tìm thêm, để trống nếu đủ rồi"
}}
"""
    else:
        return f"""
You are an AI assistant that evaluates whether the provided context is sufficient to answer a question.

Question:
{question}

Current context:
{context}

Task:
- Determine if the context is sufficient to answer the question.
- If NOT sufficient, suggest a better query to retrieve more relevant information.

Rules:
- Return ONLY valid JSON (no explanation, no extra text).
- Keep the reason concise.

Output format:
{{
  "sufficient": true or false,
  "reason": "brief explanation",
  "refined_query": "improved search query if needed, otherwise empty string"
}}
"""