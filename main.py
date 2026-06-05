from __future__ import annotations

import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv
from src.chunking import RecursiveChunker
from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
import re

def clean_legal_text(text: str) -> str:
    # 1. Xóa các dòng rác (header, footer, số trang)
    # Ví dụ: xóa các dòng như "Trang 1/5", "Bộ Tư pháp", "CỘNG HÒA XÃ HỘI..."
    # Bạn cần tùy chỉnh regex theo file luật thực tế của bạn
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Bỏ qua các dòng quá ngắn hoặc là số trang
        if not stripped or len(stripped) < 3 or re.match(r'^(Trang \d+|\d+/\d+)$', stripped):
            continue
        cleaned_lines.append(stripped)
    
    # 2. Gộp dòng: Loại bỏ các ký tự xuống dòng ngắt quãng không cần thiết
    # Đảm bảo các đoạn văn bản liền mạch trước khi chia chunk
    return "\n".join(cleaned_lines)

# SAMPLE_FILES = [
#     "data/python_intro.txt",
#     "data/vector_store_notes.md",
#     "data/rag_system_design.md",
#     "data/customer_support_playbook.txt",
#     "data/chunking_experiment_report.md",
#     "data/vi_retrieval_notes.md",
# ]

SAMPLE_FILES = [
    "data/kehoach199.txt",
    "data/luatchuyendoiso2025.txt",
    "data/luatthihanhandansu2025.txt",
    "data/nghidinh161-2026.txt",
    "data/thongtu29-2026.txt",
]
BENCHMARK_DATA = [
    {
        "query": "Luật Chuyển đổi số 2025 quy định phạm vi điều chỉnh như thế nào?",
        "gold_answer": "Luật quy định về chuyển đổi số, bao gồm nguyên tắc, chính sách, điều phối quốc gia, biện pháp bảo đảm, Chính phủ số, kinh tế số, xã hội số, và trách nhiệm của cơ quan, tổ chức, cá nhân trong chuyển đổi số."
    },
    {
        "query": "Nghị định 161/2026 quy định mức lương cơ sở từ ngày nào và là bao nhiêu?",
        "gold_answer": "Từ ngày 01/7/2026, mức lương cơ sở là 2.530.000 đồng/tháng."
    },
    {
        "query": "Thông tư 29/2026 điều chỉnh những nội dung chính nào của thị trường bán buôn điện cạnh tranh?",
        "gold_answer": "Thông tư quy định đăng ký tham gia thị trường điện, lập kế hoạch vận hành, cơ chế chào giá, lập lịch huy động, đo đếm điện năng, xác định giá thị trường và thanh toán, công bố thông tin, giám sát vận hành, và trách nhiệm của các đơn vị tham gia thị trường điện."
    },
    {
        "query": "Luật Thi hành án dân sự 2025 quy định ai có thẩm quyền giải quyết khiếu nại lần hai?",
        "gold_answer": "Thủ trưởng cơ quan quản lý thi hành án dân sự thuộc Bộ Tư pháp giải quyết khiếu nại lần hai đối với quyết định giải quyết khiếu nại chưa có hiệu lực thi hành của Thủ trưởng cơ quan thi hành án dân sự tỉnh, thành phố và của Trưởng văn phòng thi hành án dân sự."
    },
    {
        "query": "Kế hoạch 199/KH-UBND năm 2026 hướng tới mục tiêu tổng quát nào?",
        "gold_answer": "Huy động sức mạnh tổng hợp của hệ thống chính trị và toàn dân tham gia phòng, chống tội phạm và tệ nạn ma túy; từng bước xây dựng và duy trì bền vững xã, phường không ma túy trong giai đoạn 2026-2030, hướng tới xây dựng tỉnh không ma túy."
    }
]

# def load_documents_from_files(file_paths: list[str]) -> list[Document]:
#     """Load documents from file paths for the manual demo."""
#     allowed_extensions = {".md", ".txt"}
#     documents: list[Document] = []

#     for raw_path in file_paths:
#         path = Path(raw_path)

#         if path.suffix.lower() not in allowed_extensions:
#             print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
#             continue

#         if not path.exists() or not path.is_file():
#             print(f"Skipping missing file: {path}")
#             continue

#         content = path.read_text(encoding="utf-8")
#         documents.append(
#             Document(
#                 id=path.stem,
#                 content=content,
#                 metadata={"source": str(path), "extension": path.suffix.lower()},
#             )
#         )

#     return documents

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    
    # Chunker vẫn giữ nguyên
    legal_separators = ["\nĐiều ", "\nKhoản ", "\n\n", ". ", " "]
    chunker = RecursiveChunker(separators=legal_separators, chunk_size=800)

    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists(): continue

        # Đọc và Làm sạch ngay tại đây
        raw_content = path.read_text(encoding="utf-8")
        clean_content = clean_legal_text(raw_content) 
        
        # Chia chunk trên nội dung đã sạch
        chunks = chunker.chunk(clean_content)
        
        for i, chunk_text in enumerate(chunks):
            # Tinh chỉnh metadata: Cố gắng lấy số Điều nếu có thể
            # Ví dụ: lấy ra "Điều 1", "Điều 2" để làm metadata filter
            match = re.search(r'Điều (\d+)', chunk_text)
            dieu_id = match.group(1) if match else "unknown"

            documents.append(
                Document(
                    id=f"{path.stem}_chunk_{i}",
                    content=f"[Văn bản: {path.stem} | Điều: {dieu_id}]\n{chunk_text}",
                    metadata={
                        "source": str(path),
                        "doc_id": path.stem,
                        "dieu": dieu_id # Metadata này cực giá trị để filter
                    },
                )
            )
    debug_data = []
    for doc in documents:
        debug_data.append({
            "id": doc.id,
            "source": doc.metadata['source'],
            "content": doc.content
        })

    with open("debug_chunks.json", "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=4)
    print("\n[INFO] Đã lưu thông tin chunk vào file debug_chunks.json để kiểm tra!")
    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def call_openai_llm(prompt: str) -> str:
    """A real LLM caller using OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI() 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 
        )
        return response.choices[0].message.content
    except ImportError:
        return "[ERROR] Bạn chưa cài thư viện openai. Hãy chạy: pip install openai"
    except Exception as e:
        return f"[ERROR] Lỗi gọi API: {e}"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."
    print(sample_files)
    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    # Load file .env
    load_dotenv(override=False)
    
    # 1. Setup Embedder
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=5)
    
    # Lấy text an toàn dựa trên format trả về của in-memory/chroma
    for index, result in enumerate(search_results, start=1):
        content_preview = result.get('text', result.get('content', ''))[:120].replace('\n', ' ')
        print(f"{index}. score={result.get('distance', result.get('score', 0)):.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {content_preview}...")

    # 2. Setup LLM
    llm_provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if llm_provider == "openai":
        llm_fn = call_openai_llm
        print("\nLLM backend: OpenAI (gpt-3.5-turbo)")
    else:
        llm_fn = demo_llm
        print("\nLLM backend: Mock LLM")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    # print(f"Question: {query}")
    # print("Agent answer:")
    # print(agent.answer(query, top_k=3))
    # print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    
    # --- XÓA HOẶC COMMENT ĐOẠN CŨ NÀY ---
    # print(f"Question: {query}")
    # print("Agent answer:")
    # print(agent.answer(query, top_k=3))
    
    # --- THÊM DÒNG NÀY VÀO ĐỂ CHẠY BENCHMARK ---
    run_benchmark_evaluation(agent)
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)

def run_benchmark_evaluation(agent: KnowledgeBaseAgent):
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU CHẠY BENCHMARK ĐÁNH GIÁ CHIẾN LƯỢC RAG")
    print("="*60)
    
    for i, item in enumerate(BENCHMARK_DATA, start=1):
        query = item["query"]
        gold = item["gold_answer"]
        
        print(f"\n[Câu hỏi {i}]: {query}")
        
        # Gọi Agent trả lời
        agent_answer = agent.answer(query, top_k=3)
        
        print("-" * 40)
        print("RAG Trả lời   :", agent_answer.strip())
        print("Đáp án chuẩn :", gold)
        print("-" * 40)
        
        # Tạm dừng 2 giây giữa các câu hỏi để tránh bị lỗi rate-limit của API (nếu dùng API miễn phí)
        import time
        time.sleep(2)
if __name__ == "__main__":
    raise SystemExit(main())