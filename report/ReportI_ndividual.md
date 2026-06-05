# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thị Yến
**ID:** B22DCCN928
**Nhóm:** 5 ngón tay
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là hai vector biểu diễn (embedding vectors) có hướng gần như trùng nhau trong không gian vector đa chiều, thể hiện hai đoạn văn bản có sự tương đồng cao về mặt ý nghĩa ngữ nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "Luật doanh nghiệp quy định về tư cách pháp nhân."
- Sentence B: "Tư cách pháp nhân là nội dung được quy định trong Luật doanh nghiệp."

**Ví dụ LOW similarity:**
- Sentence A: "Luật doanh nghiệp."
- Sentence B: "Hướng dẫn làm món phở bò."

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo hướng (góc), không đo độ dài (magnitude). Điều này giúp so sánh sự tương đồng về ý nghĩa giữa các tài liệu có độ dài khác nhau một cách chính xác, trong khi Euclidean distance dễ bị nhiễu bởi văn bản dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(22.11) = 23` chunks.

**Tại sao muốn overlap nhiều hơn?**
> Overlap giúp giữ ngữ cảnh ở biên đoạn chia, tránh làm đứt gãy ý nghĩa của câu hoặc mệnh đề khi bị cắt ngang.

---

## 2. Document Selection — Nhóm (10 điểm)

**Domain:** Luật pháp Việt Nam.

**Metadata Schema:**
| Trường metadata | Kiểu | Ví dụ | Tại sao hữu ích? |
|----------------|------|-------|-------------------|
| `doc_id` | `str` | `luatthihanhandansu` | Dùng để filter văn bản, tăng precision. |
| `source` | `str` | `data/law.txt` | Trích dẫn nguồn. |

---

## 3. Chunking Strategy — Cá nhân chọn (15 điểm)

### So Sánh Strategy

| Tài liệu | Strategy | Chunk Count | Retrieval Quality |
|-----------|----------|-------------|-------------------|
| Luật THADS | Baseline (FixedSize) | 180 | Thấp (Hay mất ngữ cảnh) |
| Luật THADS | **Của tôi (Legal-Structure)** | 138 | Cao (Giữ trọn Điều) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------|-----------|----------|
| **Tôi** | Legal-Structure | 7/10 | Giữ cấu trúc Điều, Khoản | Cần lọc rác dữ liệu tốt hơn |
| **Vũ** | Ensemble + Rerank | 10/10 | Reranking giúp hit 5/5 | Cài đặt phức tạp |
| **Sơn** | HybridLegal + Filter | 9/10 | Metadata Filter cực mạnh | Phụ thuộc vào metadata sạch |

**Strategy nào tốt nhất?**
> Chiến lược **Ensemble + Rerank** của Vũ là tốt nhất vì nó kết hợp cả vector search và lexical reranking, đảm bảo độ chính xác vượt trội trên mọi query. Chiến lược của tôi tốt ở khâu cấu trúc (Structure), nhưng cần bổ sung bộ lọc (Filter) và Rerank để đạt kết quả tối đa.

---

## 4. My Approach — Cá nhân (10 điểm)

* **Chunking:** `RecursiveChunker` ưu tiên tách theo cấu trúc `\nĐiều`. Tôi đã thêm "Context Enrichment" (đính kèm tên văn bản vào chunk).
* **KnowledgeBaseAgent:** Dùng CoT Prompt (Trích dẫn -> Lập luận -> Kết luận) để hạn chế ảo giác.
* **Test Results:** `12/12 tests passed`.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Phạt tiền 5 triệu" | "Xử phạt 5.000.000đ" | High | 0.85 | Có |
| 2 | "Luật kinh doanh" | "Nấu ăn ngon" | Low | 0.15 | Có |

---

## 6. Results — Cá nhân (10 điểm)

| # | Query | Gold Answer | Agent Answer | Score | Relevant? |
|---|-------|-------------|--------------|-------|-----------|
| 1 | Luật CĐS 2025... | Quy định nguyên tắc... | Đúng | 0.88 | Có |
| 2 | Nghị định 161... | 2.530.000đ | Đúng | 0.92 | Có |
| 3 | Thông tư 29... | Nội dung chính... | Đúng | 0.85 | Có |
| 4 | Luật THADS 2025... | Thẩm quyền... | Sai (Sai đoạn) | 0.65 | Không |
| 5 | Kế hoạch 199... | Mục tiêu... | Không có dữ liệu | N/A | Không |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm)

**Điều hay nhất tôi học được từ thành viên khác:**
> Từ Vũ, tôi học được sức mạnh của **Lexical Reranking** (kiểm tra lại kết quả bằng từ khóa) để tăng precision. Từ Sơn, tôi thấy rằng **Metadata Filtering** (`doc_id`) là cách nhanh nhất để chặn các "nhiễu" từ tài liệu không liên quan.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> 1. Tiền xử lý (Cleaning): Xóa bỏ các boilerplate (Quốc hiệu, Tiêu ngữ) trước khi index để embedding model tập trung vào nội dung pháp lý.
> 2. Áp dụng Hybrid Search: Kết hợp Vector Search với từ khóa để giải quyết triệt để lỗi tìm sai tài liệu ở câu 4 và 5.

## Tự Đánh Giá
* **Tổng điểm:** 97 / 100