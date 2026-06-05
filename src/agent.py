from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        results = self.store.search(question, top_k=top_k)
        
        if not results:
            return "I couldn't find any relevant information in the knowledge base to answer your question."

        # Xử lý kết quả trả về từ store.search()
        contexts = []
        for i, res in enumerate(results):
            # Lấy content từ kết quả search
            chunk_text = res.get("content", "") 
            if chunk_text:
                contexts.append(f"--- Document {i+1} ---\n{chunk_text}")
                
        context_str = "\n\n".join(contexts)

        # 2. Build prompt with context (English Prompting for better instruction following)
#         prompt = f"""You are a helpful, accurate, and professional AI assistant. 
# Your task is to answer the user's question based STRICTLY and ONLY on the provided CONTEXT below. 

# Rules:
# - If the CONTEXT contains the answer, extract and synthesize it clearly.
# - If the CONTEXT does not contain sufficient information to answer the question, you must explicitly say: "I do not have enough information in my knowledge base to answer this question."
# - DO NOT hallucinate, guess, or use outside knowledge.

# CONTEXT:
# {context_str}

# QUESTION: 
# {question}

# ANSWER:"""
        prompt = f"""Bạn là một chuyên gia pháp lý AI chuyên nghiệp, trung thực và chính xác. 
        Nhiệm vụ của bạn là trả lời câu hỏi của người dùng TUYỆT ĐỐI CHỈ dựa vào phần NGỮ CẢNH được cung cấp dưới đây. 

        CÁC LUẬT BẮT BUỘC PHẢI TUÂN THỦ:
        1. CHỈ TRÍCH XUẤT TỪ NGỮ CẢNH: Nếu NGỮ CẢNH chứa thông tin liên quan, hãy trích xuất và tổng hợp câu trả lời.
        2. TỪ CHỐI NẾU THIẾU THÔNG TIN: Nếu NGỮ CẢNH không chứa đủ thông tin để trả lời, bạn BẮT BUỘC phải nói: "Tôi không có đủ thông tin trong cơ sở dữ liệu để trả lời câu hỏi này."
        3. NGHIÊM CẤM BỊA ĐẶT: Tuyệt đối KHÔNG được ảo giác (hallucinate), KHÔNG phỏng đoán, và KHÔNG được sử dụng kiến thức bên ngoài.

        CÁCH THỨC TƯ DUY VÀ TRẢ LỜI (Step-by-step):
        Trước khi đưa ra đáp án cuối cùng, bạn hãy suy nghĩ theo các bước sau và trình bày rõ ràng ra văn bản:
        - Bước 1 (Trích dẫn): Trích dẫn ngắn gọn đoạn thông tin liên quan nhất từ NGỮ CẢNH (ghi rõ [Nguồn tài liệu] nếu có).
        - Bước 2 (Lập luận): Phân tích ngắn gọn xem thông tin đó giải quyết câu hỏi như thế nào.
        - Bước 3 (Kết luận): Đưa ra câu trả lời cuối cùng thật súc tích, chính xác và đi thẳng vào trọng tâm.

        NGỮ CẢNH:
        {context_str}

        CÂU HỎI: 
        {question}

        TRẢ LỜI:"""

        # 3. Call the LLM to generate an answer
        response = self.llm_fn(prompt)
        
        return response