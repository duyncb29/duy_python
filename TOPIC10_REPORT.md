# BÁO CÁO KẾT QUẢ CẢI THIỆN RETRIEVAL TRONG RAG (TOPIC 10)

## 1. Tổng Quan Bài Tập & Bối Cảnh
Trong bài tập Topic 10, hệ thống RAG được nâng cấp từ Topic 9 nhằm khắc phục triệt để các nhược điểm của phương pháp truy xuất truyền thống (Vector-only search trên naive chunks), đặc biệt là các trường hợp:
- **Semantic-Gap**: Câu hỏi của người dùng diễn đạt khác biệt hoàn toàn với từ ngữ trong tài liệu.
- **Keyword / Code & Mã riêng**: Câu hỏi chứa các mã lỗi (`E1`), tên riêng (`SmartChef X`, `SmartLife`), thông số kỹ thuật (`2.4GHz`, `200 độ C`, `85-90 độ C`, `1900-8198`).

---

## 2. 7 Yêu Cầu Bắt Buộc & Giải Pháp Triển Khai

| STT | Yêu Cầu Bắt Buộc | Giải Pháp Triển Khai Trong Mã Nguồn |
|:---|:---|:---|
| **1** | **Bộ test (10-30 câu hỏi)** | Thiết lập `test_dataset_topic10.json` gồm 16 câu hỏi benchmark chi tiết kèm nhãn Ground Truth (Expected Chunk IDs, Expected Keywords, Reference Answer). |
| **2** | **Đo Baseline** | Đã đo đạc hệ thống Baseline (Topic 9: Naive Fixed Chunking, Vector-Only Cosine Search, Top-2). |
| **3** | **Kỹ thuật 1: Structure-Aware Chunking** | Bổ sung ngữ cảnh Metadata `[Danh mục: ...] [Tiêu đề: ...]` trực tiếp vào từng chunk text, giúp mô hình giữ trọn vẹn ngữ cảnh tiêu đề. |
| **4** | **Kỹ thuật 2: Hybrid Search & Reranking** | Tích hợp **BM25 Okapi** (Keyword Search) + **Vector Cosine Similarity** (Semantic Search) + thuật toán **Reciprocal Rank Fusion (RRF)** để rerank top candidate chunks. |
| **5** | **Đổi từng kỹ thuật, đo lại** | Thực hiện đo độc lập 4 cấu hình: (1) Baseline, (2) Tech 1, (3) Tech 2, (4) Combined (Tech 1 + Tech 2). |
| **6** | **Bảng so sánh Trước/Sau** | Đo đạc 5 chỉ số định lượng: Hit Rate @2, MRR, Semantic Hit %, Keyword/Code Hit %, Answer Accuracy %. |
| **7** | **Kết luận & Đánh giá** | Phân tích sâu kỹ thuật nào giúp nâng cao chỉ số nào nhiều nhất cho dữ liệu sản phẩm. |

---

## 3. Bảng Kết Quả Benchmark Chi Tiết (Trước & Sau Cải Tiến)

| Cấu Hình RAG | Hit Rate @2 (%) | MRR | Semantic Hit (%) | Keyword / Code Hit (%) | Độ Chính Xác Đáp Án (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Vector-Only, Naive)** | 81.2% | 0.719 | 75.0% | 87.5% | 60.9% |
| **Kỹ thuật 1 (Structure-Aware Chunking)** | 81.2% | 0.781 | 75.0% | 87.5% | 73.4% |
| **Kỹ thuật 2 (Hybrid BM25 + Vector + RRF)** | 93.8% | 0.875 | 87.5% | 100.0% | 79.7% |
| **Kết Hợp (Tech 1 + Tech 2)** | **100.0%** | **0.938** | **100.0%** | **100.0%** | **85.9%** |

---

## 4. Phân Tích Kỹ Thuật Nào Giúp Bao Nhiêu?

1. **Tác động của Kỹ thuật 1 (Structure-Aware Chunking)**:
   - Tăng **MRR** từ `0.719` lên `0.781` (+8.6%).
   - Tăng **Độ chính xác đáp án từ LLM** từ `60.9%` lên `73.4%` (+12.5%).
   - *Nguyên nhân*: Việc gắn thêm Metadata Tiêu đề & Section giúp LLM hiểu chính xác nguồn gốc tài liệu, tránh tình trạng LLM trả lời mơ hồ hoặc thiếu ngữ cảnh.

2. **Tác động của Kỹ thuật 2 (Hybrid BM25 + Vector + RRF)**:
   - Tăng **Hit Rate @2** từ `81.2%` lên `93.8%` (+12.6%).
   - Nâng tỷ lệ tìm chính xác câu hỏi Keyword/Mã riêng lên **100.0%**.
   - *Nguyên nhân*: BM25 bù đắp điểm yếu của Vector Search khi tìm kiếm các ký tự đặc biệt, mã hiệu `E1`, số điện thoại `1900-8198`, email `support@smartchef.vn`.

3. **Tác động của Phương Pháp Kết Hợp (Combined Strategy)**:
   - Đạt hiệu năng tối ưu tuyệt đối: **Hit Rate @2 = 100%**, **MRR = 0.938**, **Answer Accuracy = 85.9%**.
   - Khắc phục hoàn toàn 100% các ca Semantic Gap lẫn các ca có từ khóa/mã riêng đặc thù.

---

## 5. Hướng Dẫn Chạy Benchmark

Di chuyển vào thư mục `llm-app` và chạy lệnh:

```bash
cd llm-app
python -m uv run python topic10_retrieval_optimization.py
```
