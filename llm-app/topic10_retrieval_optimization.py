import asyncio
import json
import os
import sys
from typing import cast

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import Settings

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()


# ---------------------------------------------------------------------------
# 1. DATA MODELS
# ---------------------------------------------------------------------------
class RawDocument(BaseModel):
    """Raw document structure before chunking."""

    doc_id: int = Field(..., description="Document ID")
    category: str = Field(..., description="Category or Section name")
    title: str = Field(..., description="Document Title")
    content: str = Field(..., description="Full text content of document")


class Chunk(BaseModel):
    """Structured Chunk representation."""

    id: int = Field(..., description="Chunk ID")
    doc_id: int = Field(..., description="Parent Document ID")
    title: str = Field(..., description="Chunk or Document Title")
    text: str = Field(..., description="Chunk text content for embedding/search")
    raw_text: str = Field(..., description="Original raw text without metadata")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding of chunk"
    )


class BenchmarkCase(BaseModel):
    """Test case schema from benchmark dataset."""

    id: int
    category: str
    query: str
    expected_chunk_ids: list[int]
    expected_keywords: list[str]
    ground_truth_answer: str


class EvaluationResult(BaseModel):
    """Single evaluation result for a query under a specific configuration."""

    query_id: int
    category: str
    query: str
    config_name: str
    retrieved_chunk_ids: list[int]
    hit: bool
    reciprocal_rank: float
    answer: str
    answer_score: float


# ---------------------------------------------------------------------------
# 2. RAW KNOWLEDGE BASE DATA (SmartChef X Product Manual)
# ---------------------------------------------------------------------------
RAW_DOCUMENTS = [
    RawDocument(
        doc_id=1,
        category="Tổng quan",
        title="Giới thiệu chung về SmartChef X",
        content="Thiết bị SmartChef X là nồi đa năng thông minh tích hợp hơn 15 chế độ nấu tự động, màn hình cảm ứng OLED sắc nét và hỗ trợ kết nối Wifi băng tần 2.4GHz để điều khiển từ xa.",
    ),
    RawDocument(
        doc_id=2,
        category="Kết nối & Ứng dụng",
        title="Hướng dẫn kết nối Wifi cho SmartChef X",
        content="Để kết nối SmartChef X với mạng Wifi, trước hết hãy nhấn giữ nút Wifi trên bảng điều khiển trong 5 giây cho đến khi đèn báo nháy nhanh. Tiếp theo, mở ứng dụng SmartLife trên điện thoại, chọn Thêm thiết bị và làm theo hướng dẫn kết nối trên ứng dụng.",
    ),
    RawDocument(
        doc_id=3,
        category="Vận hành an toàn",
        title="Chế độ nấu áp suất an toàn",
        content="Khi sử dụng chế độ nấu áp suất của SmartChef X, van xả áp phải luôn ở vị trí đóng (Sealing). Tuyệt đối không cố gắng mở nắp nồi khi cột chỉ thị áp suất màu đỏ vẫn đang nổi lên. Hãy đợi nồi tự hạ áp suất hoặc nhấn nút xả áp thủ công trước khi mở.",
    ),
    RawDocument(
        doc_id=4,
        category="Vệ sinh & Bảo quản",
        title="Vệ sinh lòng nồi và khay nước ngưng tụ",
        content="Lòng nồi của SmartChef X được phủ lớp chống dính gốm cao cấp. Hãy vệ sinh lòng nồi bằng nước ấm, xà phòng dịu nhẹ và bọt biển mềm. Không dùng búi sắt hoặc chất tẩy rửa mạnh. Khay chứa nước ngưng tụ ở mặt sau cần tháo và đổ nước sau mỗi lần nấu.",
    ),
    RawDocument(
        doc_id=5,
        category="Chính sách bảo hành",
        title="Chính sách bảo hành chính hãng",
        content="Thiết bị SmartChef X được bảo hành chính hãng 24 tháng đối với các lỗi phần cứng phát sinh từ phía nhà sản xuất (như hỏng bảng điều khiển, lỗi cảm biến nhiệt). Các phụ kiện đi kèm như muỗng, xửng hấp được bảo hành 12 tháng.",
    ),
    RawDocument(
        doc_id=6,
        category="Đổi trả & Hỗ trợ",
        title="Chính sách đổi trả sản phẩm",
        content="Khách hàng được quyền đổi mới sản phẩm SmartChef X miễn phí trong vòng 7 ngày đầu kể từ ngày mua nếu sản phẩm gặp lỗi phần cứng kỹ thuật được xác nhận bởi trung tâm bảo hành. Sản phẩm đổi trả phải đầy đủ hộp và phụ kiện đi kèm.",
    ),
    RawDocument(
        doc_id=7,
        category="Xử lý sự cố",
        title="Mã lỗi E1 và cách khắc phục",
        content="Lỗi E1 hiển thị trên màn hình SmartChef X cảnh báo tình trạng nồi bị quá nhiệt (nhiệt độ lòng nồi vượt mức 200 độ C do thiếu nước hoặc bị cháy khét đáy). Cách xử lý: Rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, thêm nước trước khi nấu tiếp.",
    ),
    RawDocument(
        doc_id=8,
        category="Chế độ nấu",
        title="Chế độ nấu chậm (Slow Cook)",
        content="Chế độ Slow Cook của SmartChef X duy trì nhiệt độ ổn định ở mức 85-90 độ C trong thời gian dài (từ 2 đến 8 giờ tùy cài đặt). Chế độ này lý tưởng cho các món hầm xương, kho cá giúp giữ trọn vẹn hương vị và dưỡng chất.",
    ),
    RawDocument(
        doc_id=9,
        category="Công thức nấu ăn",
        title="Tải thêm công thức nấu ăn mới",
        content="Bạn có thể tải thêm hàng trăm công thức nấu ăn miễn phí thông qua kho công thức trực tuyến trên ứng dụng SmartLife. Các công thức mới được cập nhật tự động định kỳ vào ngày 1 hàng tháng.",
    ),
    RawDocument(
        doc_id=10,
        category="Liên hệ & Hỗ trợ",
        title="Thông tin liên hệ hỗ trợ kỹ thuật",
        content="Mọi thắc mắc kỹ thuật về SmartChef X xin vui lòng liên hệ tổng đài chăm sóc khách hàng 1900-8198 (hoạt động từ 8h00 đến 21h00 tất cả các ngày trong tuần) hoặc gửi email trực tiếp tới support@smartchef.vn.",
    ),
]


# ---------------------------------------------------------------------------
# 3. CHUNKING STRATEGIES (Baseline vs. Tech 1 Structure-Aware)
# ---------------------------------------------------------------------------
def build_baseline_chunks(docs: list[RawDocument]) -> list[Chunk]:
    """Baseline Chunking: Plain text without title/header context (Naive Chunking)."""
    chunks = []
    for doc in docs:
        chunks.append(
            Chunk(
                id=doc.doc_id,
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.content,  # Naive content only
                raw_text=doc.content,
            )
        )
    return chunks


def build_structure_aware_chunks(docs: list[RawDocument]) -> list[Chunk]:
    """Technique 1: Structure-Aware & Metadata Enriched Chunking.

    Prepends Document Title, Category Path and explicit Headers directly into the chunk text.
    This solves semantic disconnect when text mentions 'nồi' without specifying 'SmartChef X'.
    """
    chunks = []
    for doc in docs:
        enriched_text = f"[Danh mục: {doc.category}] [Tiêu đề: {doc.title}]\nNội dung: {doc.content}"
        chunks.append(
            Chunk(
                id=doc.doc_id,
                doc_id=doc.doc_id,
                title=doc.title,
                text=enriched_text,  # Contextually enriched
                raw_text=doc.content,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# 4. EMBEDDING GENERATION (Deterministic TF-IDF Hash or Real API)
# ---------------------------------------------------------------------------
def generate_deterministic_embedding(text: str) -> list[float]:
    """Generate a 256-dimensional normalized BoW vector."""
    words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
    vector = np.zeros(256, dtype=np.float32)
    for w in words:
        if len(w) > 1:
            idx = hash(w) % 256
            vector[idx] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


async def embed_chunks(
    chunks: list[Chunk], settings: Settings
) -> tuple[list[Chunk], str]:
    """Embed chunks using real Gemini/OpenAI API or fallback to mock vectorizer."""
    texts = [c.text for c in chunks]

    gemini_key = settings.gemini_api_key
    openai_key = settings.openai_api_key
    has_gemini = bool(gemini_key and gemini_key.get_secret_value())
    has_openai = bool(openai_key and openai_key.get_secret_value())

    if has_gemini and gemini_key is not None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key.get_secret_value())  # type: ignore[attr-defined]
            res = genai.embed_content(  # type: ignore[attr-defined]
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document",
            )
            embs = cast(list[list[float]], res.get("embedding", []))
            if embs:
                for i, emb in enumerate(embs):
                    chunks[i].embedding = emb
                return chunks, "Gemini API (text-embedding-004)"
        except Exception:
            pass

    if has_openai and openai_key is not None:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key.get_secret_value())
            oai_emb_res = await client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            for i, item in enumerate(oai_emb_res.data):
                chunks[i].embedding = item.embedding
            return chunks, "OpenAI API (text-embedding-3-small)"
        except Exception:
            pass

    # Fallback to deterministic mock vectorizer
    for chunk in chunks:
        chunk.embedding = generate_deterministic_embedding(chunk.text)
    return chunks, "Deterministic BoW Vectorizer (Offline Mode)"


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity score."""
    np_a = np.array(vec_a)
    np_b = np.array(vec_b)
    dot = np.dot(np_a, np_b)
    norm_a = np.linalg.norm(np_a)
    norm_b = np.linalg.norm(np_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# 5. RETRIEVAL ENGINES (Vector-Only, Hybrid BM25, RRF & Reranker)
# ---------------------------------------------------------------------------
def tokenize_vietnamese(text: str) -> list[str]:
    """Basic tokenizer for BM25 keyword matching."""
    cleaned = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(":", " ")
        .replace("-", " ")
    )
    return [w for w in cleaned.split() if len(w) > 0]


class HybridRetriever:
    """Hybrid Retrieval combining Vector Similarity & BM25 with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize_vietnamese(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search_vector(
        self, query_emb: list[float], top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Vector similarity search."""
        scores = []
        for c in self.chunks:
            if c.embedding:
                score = compute_cosine_similarity(query_emb, c.embedding)
                scores.append((c, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """BM25 keyword search."""
        tokens = tokenize_vietnamese(query)
        bm25_scores = self.bm25.get_scores(tokens)
        results = []
        for i, score in enumerate(bm25_scores):
            results.append((self.chunks[i], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_hybrid_rrf(
        self,
        query: str,
        query_emb: list[float],
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid Search combining Vector & BM25 using Reciprocal Rank Fusion (RRF)."""
        vec_results = self.search_vector(query_emb, top_k=10)
        bm25_results = self.search_bm25(query, top_k=10)

        rrf_scores: dict[int, float] = {c.id: 0.0 for c in self.chunks}

        # Accumulate Vector RRF points
        for rank, (chunk, _) in enumerate(vec_results, 1):
            rrf_scores[chunk.id] += 1.0 / (rrf_k + rank)

        # Accumulate BM25 RRF points
        for rank, (chunk, _) in enumerate(bm25_results, 1):
            rrf_scores[chunk.id] += 1.0 / (rrf_k + rank)

        # Sort chunks by fused RRF score
        chunk_map = {c.id: c for c in self.chunks}
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids[:top_k]]


# ---------------------------------------------------------------------------
# 6. LLM GENERATION & ANSWER EVALUATION
# ---------------------------------------------------------------------------
async def generate_rag_answer(
    query: str, retrieved_chunks: list[Chunk], settings: Settings
) -> str:
    """Generate answer from LLM given retrieved context."""
    context_str = "\n\n".join(
        f"--- Tài liệu {c.id} ({c.title}) ---\n{c.raw_text}" for c in retrieved_chunks
    )

    system_prompt = (
        "Bạn là trợ lý AI chuyên gia hỗ trợ sản phẩm SmartChef X.\n"
        "Hãy trả lời câu hỏi dựa TRỰC TIẾP trên phần NGỮ CẢNH tài liệu dưới đây.\n"
        "Nếu không có thông tin trong tài liệu, trả lời: 'Không tìm thấy thông tin phù hợp trong tài liệu.'"
    )

    user_prompt = f"NGỮ CẢNH:\n{context_str}\n\nCÂU HỎI: {query}"

    gemini_key = settings.gemini_api_key
    openai_key = settings.openai_api_key

    if gemini_key and gemini_key.get_secret_value():
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key.get_secret_value())  # type: ignore[attr-defined]
            model = genai.GenerativeModel(  # type: ignore[attr-defined]
                model_name="gemini-1.5-flash", generation_config={"temperature": 0.0}
            )
            res = await model.generate_content_async(
                f"{system_prompt}\n\n{user_prompt}"
            )
            return str(res.text).strip()
        except Exception:
            pass

    if openai_key and openai_key.get_secret_value():
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key.get_secret_value())
            oai_chat_res = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return oai_chat_res.choices[0].message.content or ""
        except Exception:
            pass

    # Deterministic Mock Answer based on retrieved context
    if not retrieved_chunks:
        return "Không tìm thấy thông tin phù hợp trong tài liệu."

    best_chunk = retrieved_chunks[0]
    return f"Dựa theo tài liệu '{best_chunk.title}': {best_chunk.raw_text}"


def evaluate_answer_score(answer: str, expected_keywords: list[str]) -> float:
    """Evaluate LLM answer quality based on coverage of expected ground truth keywords."""
    if not expected_keywords:
        return 1.0

    matches = 0
    ans_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() in ans_lower:
            matches += 1

    return matches / len(expected_keywords)


# ---------------------------------------------------------------------------
# 7. BENCHMARK PIPELINE EXECUTION
# ---------------------------------------------------------------------------
async def run_benchmark() -> None:
    console.print(
        Panel(
            "[bold green]TOPIC 10: RAG RETRIEVAL OPTIMIZATION & BENCHMARK SUITE[/bold green]\n"
            "Đo đạc & So sánh: Baseline vs. Structure-Aware Chunking vs. Hybrid Search (BM25+Vector+RRF)",
            box=box.DOUBLE,
            style="cyan",
        )
    )

    settings = Settings()

    # Load Test Dataset
    dataset_path = "test_dataset_topic10.json"
    if not os.path.exists(dataset_path):
        console.print(f"[bold red]Lỗi: Tệp {dataset_path} không tồn tại![/bold red]")
        return

    with open(dataset_path, encoding="utf-8") as f:
        raw_cases = json.load(f)

    test_cases = [BenchmarkCase(**case) for case in raw_cases]
    console.print(
        f"[green]✔ Đã tải {len(test_cases)} câu hỏi kiểm thử từ bộ test benchmark.[/green]\n"
    )

    # 1. Build Baseline Chunks & Embeddings
    baseline_chunks = build_baseline_chunks(RAW_DOCUMENTS)
    baseline_chunks, emb_mode = await embed_chunks(baseline_chunks, settings)
    baseline_retriever = HybridRetriever(baseline_chunks)

    # 2. Build Structure-Aware Chunks & Embeddings (Technique 1)
    struct_chunks = build_structure_aware_chunks(RAW_DOCUMENTS)
    struct_chunks, _ = await embed_chunks(struct_chunks, settings)
    struct_retriever = HybridRetriever(struct_chunks)

    console.print(f"[green]✔ Chế độ Vector Embedding: {emb_mode}[/green]\n")

    # Configurations to evaluate:
    # 1. Baseline (Naive Chunks, Vector-Only Top-2)
    # 2. Tech 1 (Structure-Aware Chunks, Vector-Only Top-2)
    # 3. Tech 2 (Naive Chunks, Hybrid BM25+Vector+RRF Top-2)
    # 4. Combined (Structure-Aware Chunks, Hybrid BM25+Vector+RRF Top-2)
    configs = [
        ("Baseline (Vector Only, Naive)", "naive", "vector"),
        ("Kỹ thuật 1 (Structure-Aware Chunking)", "structure", "vector"),
        ("Kỹ thuật 2 (Hybrid BM25 + Vector + RRF)", "naive", "hybrid"),
        ("Kết hợp (Tech 1 + Tech 2)", "structure", "hybrid"),
    ]

    all_results: dict[str, list[EvaluationResult]] = {cfg[0]: [] for cfg in configs}

    for case in test_cases:
        # Precompute query embedding
        is_mock = "Offline" in emb_mode or "BoW" in emb_mode
        if is_mock:
            q_emb = generate_deterministic_embedding(case.query)
        else:
            from my_rag import get_embeddings

            embs, _ = await get_embeddings([case.query], settings)
            q_emb = embs[0]

        for config_name, chunk_style, search_style in configs:
            retriever = (
                struct_retriever if chunk_style == "structure" else baseline_retriever
            )

            if search_style == "vector":
                search_res = retriever.search_vector(q_emb, top_k=2)
                retrieved_chunks = [c for c, _ in search_res]
            else:  # Hybrid
                search_res = retriever.search_hybrid_rrf(
                    case.query, q_emb, top_k=2, rrf_k=60
                )
                retrieved_chunks = [c for c, _ in search_res]

            retrieved_ids = [c.id for c in retrieved_chunks]

            # Calculate Hit Rate & Reciprocal Rank
            hit = any(cid in case.expected_chunk_ids for cid in retrieved_ids)

            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in case.expected_chunk_ids:
                    rr = 1.0 / rank
                    break

            # Generate Answer & Answer Score
            ans = await generate_rag_answer(case.query, retrieved_chunks, settings)
            ans_score = evaluate_answer_score(ans, case.expected_keywords)

            eval_res = EvaluationResult(
                query_id=case.id,
                category=case.category,
                query=case.query,
                config_name=config_name,
                retrieved_chunk_ids=retrieved_ids,
                hit=hit,
                reciprocal_rank=rr,
                answer=ans,
                answer_score=ans_score,
            )
            all_results[config_name].append(eval_res)

    # ---------------------------------------------------------------------------
    # 8. PRINT BENCHMARK COMPARISON TABLE
    # ---------------------------------------------------------------------------
    table = Table(
        title="📊 BẢNG SO SÁNH HIỆU NĂNG RETRIEVAL TRƯỚC VÀ SAU CẢI TIẾN (TOPIC 10)",
        box=box.ROUNDED,
    )
    table.add_column("Cấu hình RAG", style="bold cyan")
    table.add_column("Hit Rate @2 (%)", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Semantic Hit (%)", justify="right")
    table.add_column("Keyword/Code Hit (%)", justify="right")
    table.add_column("Độ chính xác đáp án (%)", justify="right")

    summary_data = []

    for config_name, res_list in all_results.items():
        total = len(res_list)
        hits = sum(1 for r in res_list if r.hit)
        hit_rate = (hits / total) * 100

        mrr = sum(r.reciprocal_rank for r in res_list) / total

        semantic_res = [r for r in res_list if r.category == "semantic_gap"]
        semantic_hits = (
            sum(1 for r in semantic_res if r.hit) / len(semantic_res) * 100
            if semantic_res
            else 0.0
        )

        keyword_res = [r for r in res_list if r.category == "keyword_code"]
        keyword_hits = (
            sum(1 for r in keyword_res if r.hit) / len(keyword_res) * 100
            if keyword_res
            else 0.0
        )

        avg_ans_score = (sum(r.answer_score for r in res_list) / total) * 100

        summary_data.append(
            {
                "config": config_name,
                "hit_rate": hit_rate,
                "mrr": mrr,
                "semantic_hit": semantic_hits,
                "keyword_hit": keyword_hits,
                "ans_score": avg_ans_score,
            }
        )

        table.add_row(
            config_name,
            f"{hit_rate:.1f}%",
            f"{mrr:.3f}",
            f"{semantic_hits:.1f}%",
            f"{keyword_hits:.1f}%",
            f"{avg_ans_score:.1f}%",
        )

    console.print(table)

    # Save benchmark results to JSON file
    report_output = {
        "embedding_mode": emb_mode,
        "total_test_cases": len(test_cases),
        "summary": summary_data,
    }

    output_json_path = "topic10_benchmark_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report_output, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n[bold green]✔ Đã xuất dữ liệu benchmark ra file: {output_json_path}[/bold green]"
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
