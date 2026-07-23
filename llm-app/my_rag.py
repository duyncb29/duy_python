import asyncio
import sys
from typing import cast

import numpy as np
from pydantic import BaseModel, Field

from config import Settings

# Đảm bảo hiển thị tiếng Việt chính xác trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# 1. DATA MODEL DEFINITIONS (Yêu cầu bắt buộc & Pydantic Validation)
# ---------------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """Lớp dữ liệu đại diện cho một chunk tài liệu."""

    id: int = Field(..., description="ID định danh duy nhất của chunk")
    title: str = Field(..., description="Tiêu đề/Nhãn của chunk tài liệu")
    text: str = Field(..., description="Nội dung chi tiết của chunk tài liệu")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding của chunk"
    )


class RAGResponse(BaseModel):
    """Lớp dữ liệu chứa kết quả phản hồi của hệ thống RAG."""

    answer: str = Field(..., description="Câu trả lời cuối cùng từ hệ thống")
    retrieved_sources: list[tuple[int, str, float]] = Field(
        ...,
        description="Danh sách nguồn được dùng (ID, Title, Cosine Similarity Score)",
    )
    mode: str = Field(..., description="Chế độ chạy (Real API hoặc Mock Simulation)")


# ---------------------------------------------------------------------------
# 2. REAL DOCUMENT CHUNKS (Yêu cầu 1: 8-15 chunk tự đủ nghĩa)
# ---------------------------------------------------------------------------
CHUNKS_DATA = [
    DocumentChunk(
        id=1,
        title="Giới thiệu chung về SmartChef X",
        text="Thiết bị SmartChef X là nồi đa năng thông minh tích hợp hơn 15 chế độ nấu tự động, màn hình cảm ứng OLED sắc nét và hỗ trợ kết nối Wifi băng tần 2.4GHz để điều khiển từ xa.",
    ),
    DocumentChunk(
        id=2,
        title="Hướng dẫn kết nối Wifi cho SmartChef X",
        text="Để kết nối SmartChef X với mạng Wifi, trước hết hãy nhấn giữ nút Wifi trên bảng điều khiển trong 5 giây cho đến khi đèn báo nháy nhanh. Tiếp theo, mở ứng dụng SmartLife trên điện thoại, chọn Thêm thiết bị và làm theo hướng dẫn kết nối trên ứng dụng.",
    ),
    DocumentChunk(
        id=3,
        title="Chế độ nấu áp suất an toàn",
        text="Khi sử dụng chế độ nấu áp suất của SmartChef X, van xả áp phải luôn ở vị trí đóng (Sealing). Tuyệt đối không cố gắng mở nắp nồi khi cột chỉ thị áp suất màu đỏ vẫn đang nổi lên. Hãy đợi nồi tự hạ áp suất hoặc nhấn nút xả áp thủ công trước khi mở.",
    ),
    DocumentChunk(
        id=4,
        title="Vệ sinh lòng nồi và khay nước ngưng tụ",
        text="Lòng nồi của SmartChef X được phủ lớp chống dính gốm cao cấp. Hãy vệ sinh lòng nồi bằng nước ấm, xà phòng dịu nhẹ và bọt biển mềm. Không dùng búi sắt hoặc chất tẩy rửa mạnh. Khay chứa nước ngưng tụ ở mặt sau cần tháo và đổ nước sau mỗi lần nấu.",
    ),
    DocumentChunk(
        id=5,
        title="Chính sách bảo hành chính hãng",
        text="Thiết bị SmartChef X được bảo hành chính hãng 24 tháng đối với các lỗi phần cứng phát sinh từ phía nhà sản xuất (như hỏng bảng điều khiển, lỗi cảm biến nhiệt). Các phụ kiện đi kèm như muỗng, xửng hấp được bảo hành 12 tháng.",
    ),
    DocumentChunk(
        id=6,
        title="Chính sách đổi trả sản phẩm",
        text="Khách hàng được quyền đổi mới sản phẩm SmartChef X miễn phí trong vòng 7 ngày đầu kể từ ngày mua nếu sản phẩm gặp lỗi phần cứng kỹ thuật được xác nhận bởi trung tâm bảo hành. Sản phẩm đổi trả phải đầy đủ hộp và phụ kiện đi kèm.",
    ),
    DocumentChunk(
        id=7,
        title="Mã lỗi E1 và cách khắc phục",
        text="Lỗi E1 hiển thị trên màn hình SmartChef X cảnh báo tình trạng nồi bị quá nhiệt (nhiệt độ lòng nồi vượt mức 200 độ C do thiếu nước hoặc bị cháy khét đáy). Cách xử lý: Rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, thêm nước trước khi nấu tiếp.",
    ),
    DocumentChunk(
        id=8,
        title="Chế độ nấu chậm (Slow Cook)",
        text="Chế độ Slow Cook của SmartChef X duy trì nhiệt độ ổn định ở mức 85-90 độ C trong thời gian dài (từ 2 đến 8 giờ tùy cài đặt). Chế độ này lý tưởng cho các món hầm xương, kho cá giúp giữ trọn vẹn hương vị và dưỡng chất.",
    ),
    DocumentChunk(
        id=9,
        title="Tải thêm công thức nấu ăn mới",
        text="Bạn có thể tải thêm hàng trăm công thức nấu ăn miễn phí thông qua kho công thức trực tuyến trên ứng dụng SmartLife. Các công thức mới được cập nhật tự động định kỳ vào ngày 1 hàng tháng.",
    ),
    DocumentChunk(
        id=10,
        title="Thông tin liên hệ hỗ trợ kỹ thuật",
        text="Mọi thắc mắc kỹ thuật về SmartChef X xin vui lòng liên hệ tổng đài chăm sóc khách hàng 1900-8198 (hoạt động từ 8h00 đến 21h00 tất cả các ngày trong tuần) hoặc gửi email trực tiếp tới support@smartchef.vn.",
    ),
]


# ---------------------------------------------------------------------------
# 3. EMBEDDING FUNCTIONS (Yêu cầu 2, 4: Cùng model embedding, Indexing)
# ---------------------------------------------------------------------------
def generate_mock_embedding(text: str) -> list[float]:
    """Tạo vector embedding giả lập (128 dims) dựa trên tần suất từ (Bag of Words).

    Giúp hệ thống chạy thử nghiệm ngoại tuyến mượt mà mà vẫn đảm bảo tính toán cosine
    và tìm kiếm ngữ cảnh chính xác tương đối.
    """
    words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
    vector = np.zeros(128, dtype=np.float32)
    for w in words:
        if len(w) > 1:
            idx = hash(w) % 128
            vector[idx] += 1.0

    # Chuẩn hóa vector về độ dài 1 (unit norm) để cosine similarity chỉ là dot product
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


async def get_embeddings(
    texts: list[str], settings: Settings
) -> tuple[list[list[float]], str]:
    """Lấy danh sách vector embeddings bằng cách gọi API thật (Gemini/OpenAI) hoặc Mock."""
    # Narrow types and assign to local variables
    gemini_key = settings.gemini_api_key
    openai_key = settings.openai_api_key
    openrouter_key = settings.openrouter_api_key

    has_gemini = bool(gemini_key and gemini_key.get_secret_value())
    has_openai = bool(openai_key and openai_key.get_secret_value())
    has_openrouter = bool(openrouter_key and openrouter_key.get_secret_value())

    if has_gemini and gemini_key is not None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key.get_secret_value())  # type: ignore[attr-defined]
            # Gọi API lấy embedding theo lô (batch) để tối ưu chi phí và tốc độ
            gemini_response = genai.embed_content(  # type: ignore[attr-defined]
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = cast(list[list[float]], gemini_response.get("embedding", []))
            if embeddings:
                return embeddings, "Gemini API (models/text-embedding-004)"
        except Exception as e:
            console.print(
                f"[yellow]Cảnh báo lỗi Gemini Embedding: {e}. Chuyển sang chế độ Mock.[/yellow]"
            )

    if has_openai and openai_key is not None:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key.get_secret_value())
            openai_response = await client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            embeddings = [emb.embedding for emb in openai_response.data]
            return embeddings, "OpenAI API (text-embedding-3-small)"
        except Exception as e:
            console.print(
                f"[yellow]Cảnh báo lỗi OpenAI Embedding: {e}. Chuyển sang chế độ Mock.[/yellow]"
            )

    # Fallback sang Mock embedding
    mock_embs = [generate_mock_embedding(t) for t in texts]
    mode_name = "Mock Simulation Mode (TF-IDF Hash)"
    if has_openrouter:
        mode_name += " (OpenRouter detected, but OpenRouter does not support standard embeddings natively)"
    return mock_embs, mode_name


# ---------------------------------------------------------------------------
# 4. SIMILARITY SEARCH (Yêu cầu 3: Tìm top-k cosine similarity)
# ---------------------------------------------------------------------------
def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính điểm tương đồng cosine giữa 2 vector."""
    np_a = np.array(vec_a)
    np_b = np.array(vec_b)
    dot_product = np.dot(np_a, np_b)
    norm_a = np.linalg.norm(np_a)
    norm_b = np.linalg.norm(np_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def search_top_k(
    query_emb: list[float], indexed_chunks: list[DocumentChunk], k: int = 2
) -> list[tuple[DocumentChunk, float]]:
    """Tìm k chunk tài liệu có độ tương đồng cao nhất với vector truy vấn."""
    results = []
    for chunk in indexed_chunks:
        if chunk.embedding is not None:
            score = compute_cosine_similarity(query_emb, chunk.embedding)
            results.append((chunk, score))

    # Sắp xếp giảm dần theo điểm similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# 5. LLM TEXT GENERATION (Yêu cầu 5, 6: Prompt Grounding & Tránh bịa đặt)
# ---------------------------------------------------------------------------
async def call_llm_with_context(
    query: str, context_chunks: list[DocumentChunk], settings: Settings, use_mock: bool
) -> str:
    """Gọi LLM (thật hoặc mock) với prompt được thiết kế chặt chẽ (Prompt Grounding)."""

    # Ghép nội dung ngữ cảnh với delimiter rõ ràng để tránh Prompt Injection
    context_text = "\n\n".join(
        f"--- TÀI LIỆU SỐ {i + 1} (Tiêu đề: {chunk.title}) ---\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    )

    # Prompt Grounding nghiêm ngặt
    system_prompt = (
        "Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm SmartChef X.\n"
        "Nhiệm vụ duy nhất của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH tài liệu được cung cấp.\n"
        "Hãy tuân thủ nghiêm ngặt các quy định sau:\n"
        "1. CHỈ sử dụng thông tin có trong phần NGỮ CẢNH bên dưới để trả lời câu hỏi.\n"
        "2. Không được sử dụng kiến thức bên ngoài, không suy diễn thêm hoặc bịa đặt thông tin.\n"
        "3. Nếu NGỮ CẢNH không chứa thông tin cần thiết hoặc không liên quan đến câu hỏi, bạn BẮT BUỘC phải trả lời chính xác là: 'Không tìm thấy thông tin phù hợp trong tài liệu.'\n"
        "4. Câu trả lời của bạn phải ngắn gọn, súc tích và mạch lạc bằng tiếng Việt."
    )

    user_prompt = f"""Hãy đọc kỹ phần NGỮ CẢNH bên dưới và trả lời câu hỏi của người dùng.

[NGỮ CẢNH BẮT BUỘC]
<context>
{context_text}
</context>

[CÂU HỎI NGƯỜI DÙNG]
{query}
"""

    # Nếu đang chạy chế độ Mock
    if use_mock:
        await asyncio.sleep(0.5)  # Giả lập độ trễ mạng
        query_lower = query.lower()
        if "wifi" in query_lower or "kết nối" in query_lower:
            return (
                "Dựa trên tài liệu hướng dẫn, để kết nối SmartChef X với Wifi, bạn cần làm theo các bước sau:\n"
                "1. Nhấn giữ nút Wifi trên bảng điều khiển trong 5 giây cho đến khi đèn báo nháy nhanh.\n"
                "2. Mở ứng dụng SmartLife trên điện thoại, chọn 'Thêm thiết bị' và làm theo hướng dẫn kết nối để hoàn tất."
            )
        elif "e1" in query_lower or "lỗi e1" in query_lower:
            return (
                "Dựa trên tài liệu hướng dẫn, lỗi E1 cảnh báo nồi đang bị quá nhiệt (nhiệt độ lòng nồi vượt quá 200 độ C do thiếu nước hoặc cháy khét).\n"
                "Cách khắc phục: Bạn hãy rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, sau đó bổ sung nước trước khi tiếp tục nấu."
            )
        else:
            return "Không tìm thấy thông tin phù hợp trong tài liệu."

    # Chế độ chạy thật bằng API
    gemini_key = settings.gemini_api_key
    openai_key = settings.openai_api_key
    openrouter_key = settings.openrouter_api_key

    has_gemini = bool(gemini_key and gemini_key.get_secret_value())
    has_openai = bool(openai_key and openai_key.get_secret_value())
    has_openrouter = bool(openrouter_key and openrouter_key.get_secret_value())

    if has_gemini and gemini_key is not None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key.get_secret_value())  # type: ignore[attr-defined]
            # Sử dụng gemini-1.5-flash với temperature = 0.0 để kết quả ổn định và bám ngữ cảnh nhất
            model = genai.GenerativeModel(  # type: ignore[attr-defined]
                model_name="gemini-1.5-flash", generation_config={"temperature": 0.0}
            )
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            gemini_response = await model.generate_content_async(combined_prompt)
            return str(gemini_response.text).strip()
        except Exception as e:
            console.print(
                f"[red]Lỗi gọi Gemini API: {e}. Chuyển sang backup provider...[/red]"
            )

    if has_openai and openai_key is not None:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key.get_secret_value())
            openai_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                timeout=float(settings.timeout_seconds),
            )
            return openai_response.choices[0].message.content or ""
        except Exception as e:
            console.print(
                f"[red]Lỗi gọi OpenAI API: {e}. Chuyển sang backup provider...[/red]"
            )

    if has_openrouter and openrouter_key is not None:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key.get_secret_value(),
            )
            openrouter_response = await client.chat.completions.create(
                model="google/gemini-2.5-flash:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                timeout=float(settings.timeout_seconds),
            )
            return openrouter_response.choices[0].message.content or ""
        except Exception as e:
            console.print(f"[red]Lỗi gọi OpenRouter API: {e}[/red]")

    return "Không tìm thấy thông tin phù hợp trong tài liệu (Không thể kết nối các API LLM)."


# ---------------------------------------------------------------------------
# 6. PIPELINE COORDINATOR (Điều phối chính: Chunk -> Embed -> Search -> Gen)
# ---------------------------------------------------------------------------
async def run_rag_pipeline(
    query: str,
    indexed_chunks: list[DocumentChunk],
    settings: Settings,
    embedding_mode: str,
    k: int = 2,
    sim_threshold: float = 0.25,
) -> RAGResponse:
    """Quy trình RAG tích hợp đầy đủ các chốt kiểm tra."""

    # 1. Sinh vector embedding cho câu hỏi của người dùng (Yêu cầu 4)
    # Lấy nhãn xem đang ở chế độ mock hay real
    is_mock = "Mock" in embedding_mode
    if is_mock:
        query_emb = generate_mock_embedding(query)
    else:
        # Gọi hàm get_embeddings (dù là list 1 phần tử) để đồng bộ hóa model sử dụng
        embs, _ = await get_embeddings([query], settings)
        query_emb = embs[0]

    # 2. Tìm kiếm top-k (Yêu cầu 3)
    search_results = search_top_k(query_emb, indexed_chunks, k=k)

    # 3. Lọc và kiểm tra ngưỡng tương đồng (Yêu cầu 6: Đường không biết)
    # Điều chỉnh ngưỡng tùy chế độ
    actual_threshold = sim_threshold if not is_mock else 0.12

    valid_chunks = []
    retrieved_sources = []

    for chunk, score in search_results:
        retrieved_sources.append((chunk.id, chunk.title, score))
        if score >= actual_threshold:
            valid_chunks.append(chunk)

    # Nếu không có tài liệu nào đủ tương đồng, chặn ngay từ vòng gửi xe (Input Guardrail)
    if not valid_chunks:
        return RAGResponse(
            answer="Không tìm thấy thông tin phù hợp trong tài liệu.",
            retrieved_sources=retrieved_sources,
            mode=embedding_mode,
        )

    # 4. Gọi LLM sinh phản hồi bám ngữ cảnh (Yêu cầu 5, 7)
    answer = await call_llm_with_context(query, valid_chunks, settings, is_mock)

    return RAGResponse(
        answer=answer, retrieved_sources=retrieved_sources, mode=embedding_mode
    )


# ---------------------------------------------------------------------------
# 7. MAIN DEMO ENGINE (Yêu cầu 6: Chạy thử nghiệm 3 ca kiểm nghiệm)
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(
        Panel(
            "[bold cyan]AI ENGINEER ROADMAP - TOPIC 8: MINI RAG PIPELINE (NUMPY & IN-MEMORY)[/bold cyan]\n"
            "Chương trình RAG hoàn chỉnh thực hiện: Chunking -> Batch Embedding -> Cosine Similarities -> Grounding -> Citation",
            box=box.DOUBLE,
            style="cyan",
        )
    )

    # Load Settings cấu hình
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[bold red]Lỗi tải cấu hình .env:[/bold red] {e}")
        return

    # Khởi động quy trình Indexing (Yêu cầu 2: Embed toàn bộ một lần)
    console.print(
        "[yellow]>>> Bắt đầu quy trình Indexing (Tải tài liệu & Embed một lần)...[/yellow]"
    )

    chunk_texts = [c.text for c in CHUNKS_DATA]
    embeddings, emb_mode = await get_embeddings(chunk_texts, settings)

    # Lưu vector cùng text vào bộ lưu trữ in-memory
    for i, emb in enumerate(embeddings):
        CHUNKS_DATA[i].embedding = emb

    console.print(
        f"[green]✔ Indexing hoàn tất! Đã lưu trữ {len(CHUNKS_DATA)} chunks.[/green]"
    )
    console.print(f"[green]✔ Chế độ Embedding sử dụng: {emb_mode}[/green]\n")

    # Danh sách 3 ca test bắt buộc (Yêu cầu 6: test 3 ca)
    test_queries = [
        (
            "Trong kho tài liệu",
            "Làm thế nào để kết nối SmartChef X với mạng Wifi gia đình?",
        ),
        (
            "Diễn đạt khác chữ",
            "Nồi hiển thị chữ E1 trên màn hình OLED thì phải làm sao hả shop?",
        ),
        (
            "Ngoài kho tài liệu",
            "Thời tiết hôm nay ở Thành phố Hồ Chí Minh có mưa không?",
        ),
    ]

    # Chạy lần lượt các ca test
    for test_idx, (cat_name, query) in enumerate(test_queries, 1):
        console.print(
            Panel(
                f"[bold green]CA TEST {test_idx}: {cat_name}[/bold green]\n[bold]Câu hỏi:[/bold] {query}",
                box=box.ROUNDED,
                style="yellow",
            )
        )

        # Chạy pipeline RAG
        response = await run_rag_pipeline(
            query=query,
            indexed_chunks=CHUNKS_DATA,
            settings=settings,
            embedding_mode=emb_mode,
            k=2,
        )

        # In bảng kết quả tương đồng Cosine (Gợi ý 3: In điểm Cosine để trực quan hóa)
        table = Table(
            title="🔍 Điểm tương đồng Cosine của các Chunk (Top-2)", box=box.SIMPLE
        )
        table.add_column("ID Chunk", style="dim", width=10)
        table.add_column("Tiêu đề tài liệu", style="bold")
        table.add_column("Điểm số Cosine", justify="right")
        table.add_column("Trạng thái sử dụng", justify="center")

        for chunk_id, title, score in response.retrieved_sources:
            # Lấy thông tin trạng thái
            is_used = (
                "Được dùng (Context)"
                if score >= (0.25 if "Mock" not in response.mode else 0.12)
                else "Bỏ qua (Dưới ngưỡng)"
            )
            color = "green" if "Được dùng" in is_used else "red"
            table.add_row(
                str(chunk_id), title, f"{score:.4f}", f"[{color}]{is_used}[/{color}]"
            )
        console.print(table)

        # In phản hồi của LLM (Yêu cầu 5: Grounded output, Yêu cầu 7: Hiển thị nguồn rõ ràng)
        console.print("[bold cyan]▶ Phản hồi từ LLM:[/bold cyan]")
        console.print(Panel(response.answer, box=box.MINIMAL, style="bright_white"))

        # Hiển thị nguồn trích dẫn cụ thể (Yêu cầu 7: Hiện nguồn)
        used_sources = [
            f"[ID {cid}] {title}"
            for cid, title, score in response.retrieved_sources
            if score >= (0.25 if "Mock" not in response.mode else 0.12)
        ]
        if used_sources:
            console.print(
                f"[bold green]✦ Nguồn trích dẫn sử dụng:[/bold green] {', '.join(used_sources)}"
            )
        else:
            console.print(
                "[bold red]✦ Nguồn trích dẫn sử dụng:[/bold red] Không có nguồn phù hợp (Truy vấn ngoài kho)"
            )

        console.print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
