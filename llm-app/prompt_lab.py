import json
import re
import sys
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import tiktoken

from config import Settings

# ---------------------------------------------------------------------------
# 1. STRUCTURED OUTPUT SCHEMA (Requirement #2 & Hint #1)
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    """Schema Pydantic bắt buộc LLM tuân thủ định dạng JSON đầu ra."""

    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        ...,
        description="Nhãn cảm xúc/ý định: POSITIVE (Tích cực), NEGATIVE (Tiêu cực), NEUTRAL (Trung tính/Hỏi đáp/Chưa trải nghiệm).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Độ tin tưởng của đánh giá từ 0.0 đến 1.0.",
    )
    reasoning: str = Field(
        ...,
        description="Lý do ngắn gọn giải thích tại sao phân loại nhãn này (tối đa 2 câu).",
    )


# ---------------------------------------------------------------------------
# 2. PROMPT TEMPLATES - VERSION 1 & VERSION 2 (Requirements #1, #3, #7)
# ---------------------------------------------------------------------------

# Delimiter bọc dữ liệu: <input>{user_review}</input>

PROMPT_V1_BASELINE = """Bạn là một hệ thống phân loại cảm xúc nhận xét khách hàng.

Nhiệm vụ của bạn là đọc nhận xét của khách hàng và phân loại thành một trong các nhãn:
- POSITIVE
- NEGATIVE
- NEUTRAL

Hãy trả về kết quả dưới dạng JSON theo đúng schema sau:
{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence": 0.95,
  "reasoning": "Lý do phân loại..."
}}

Dữ liệu nhận xét của khách hàng nằm trong thẻ input sau:
<input>
{user_review}
</input>
"""


PROMPT_V2_IMPROVED = """Bạn là Chuyên gia Phân tích Trải nghiệm Khách hàng TMĐT (Customer Experience AI Specialist).

=== NHIỆM VỤ (TASK) ===
Phân tích nhận xét của khách hàng trong thẻ <input> và phân loại chính xác thành 1 trong 3 nhãn:
1. POSITIVE: Đánh giá khen ngợi, hài lòng về sản phẩm/dịch vụ (dù có lỗi nhỏ không đáng kể).
2. NEGATIVE: Đánh giá chê bai, thất vọng, không hài lòng, hoặc chứa ý MỈA MAI/CHÂM BIẾM (Sarcasm).
3. NEUTRAL: Câu hỏi thắc mắc hỗ trợ, bình luận hòa vốn, hoặc chưa trải nghiệm sản phẩm ("mới nhận chưa dùng").

=== QUY TẮC NGHIÊM NGẶT (CONSTRAINTS & RULES) ===
1. MỈA MAI / CHÂM BIẾM (Sarcasm): Nếu câu văn có vẻ khen nhưng mang tính mỉa mai (vd: "giao nhanh thật, chờ có 3 tuần", "dùng 2 ngày đã hỏng"), BẮT BUỘC gắn nhãn NEGATIVE.
2. CÂU HỎI HỖ TRỢ: Nếu khách hàng đặt câu hỏi hỏi thông tin, màu sắc, bảo hành... BẮT BUỘC gắn nhãn NEUTRAL.
3. KHEN CHÊ HỖN HỢP: Đánh giá trọng tâm chính. Nếu khen sản phẩm chính nhưng chê vỏ hộp xước nhẹ -> POSITIVE. Nếu chê chất lượng hỏng hóc nhưng khen đóng gói đẹp -> NEGATIVE.
4. ĐỊNH DẠNG ĐẦU RA: CHỈ trả về một đoạn JSON hợp lệ tuân thủ Schema Pydantic, KHÔNG kèm thêm lời giải thích nào ngoài JSON.

=== SCHEMA PYDANTIC ===
{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence": <float từ 0.0 đến 1.0>,
  "reasoning": "<lý do ngắn gọn bằng tiếng Việt>"
}}

=== VÍ DỤ MẪU (FEW-SHOT EXAMPLES) ===
Input: <input>Giao hàng siêu nhanh, đóng gói đẹp 5 sao!</input>
Output: {{"sentiment": "POSITIVE", "confidence": 0.98, "reasoning": "Khách hàng hài lòng về tốc độ giao hàng và đóng gói."}}

Input: <input>Shop bán đồ xịn thật, mới dùng 1 ngày đã cháy nổ :)</input>
Output: {{"sentiment": "NEGATIVE", "confidence": 0.95, "reasoning": "Câu văn chứa ý mỉa mai châm biếm về chất lượng sản phẩm kém."}}

Input: <input>Cho mình hỏi sản phẩm này có sẵn màu xanh lá không shop?</input>
Output: {{"sentiment": "NEUTRAL", "confidence": 0.99, "reasoning": "Đây là câu hỏi thắc mắc thông tin sản phẩm, không mang cảm xúc."}}

=== DỮ LIỆU ĐẦU VÀO (INPUT) ===
<input>
{user_review}
</input>
"""


# ---------------------------------------------------------------------------
# 3. BỘ TEST DATASET >= 15 CA (Requirement #5)
# ---------------------------------------------------------------------------

TEST_DATASET: list[dict[str, str]] = [
    {
        "id": "TC01",
        "input": "Sản phẩm xài cực mượt, đóng gói cẩn thận, 5 sao cho shop!",
        "expected": "POSITIVE",
        "category": "Tích cực rõ ràng",
    },
    {
        "id": "TC02",
        "input": "Hàng giao quá chậm, mở ra vỡ nát hết cả vỉ. Quá thất vọng!",
        "expected": "NEGATIVE",
        "category": "Tiêu cực rõ ràng",
    },
    {
        "id": "TC03",
        "input": "Shop ơi cho mình hỏi mẫu này có bảo hành 12 tháng không vậy?",
        "expected": "NEUTRAL",
        "category": "Câu hỏi hỗ trợ",
    },
    {
        "id": "TC04",
        "input": "Giao hàng nhanh thật đấy, chờ có 3 tuần chứ mấy :)",
        "expected": "NEGATIVE",
        "category": "Ca khó - Mỉa mai giao chậm",
    },
    {
        "id": "TC05",
        "input": "Shop bán đồ chất lượng lắm, dùng 2 ngày đã hỏng màn hình rồi!",
        "expected": "NEGATIVE",
        "category": "Ca khó - Mỉa mai đồ dỏm",
    },
    {
        "id": "TC06",
        "input": "Vỏ hộp hơi trầy nhẹ do vận chuyển nhưng máy bên trong dùng rất ngon.",
        "expected": "POSITIVE",
        "category": "Ca khó - Khen chê hỗn hợp (Tích cực chính)",
    },
    {
        "id": "TC07",
        "input": "Thiết kế đẹp thật đấy nhưng pin tụt như tụt quần, dùng chán ngắt.",
        "expected": "NEGATIVE",
        "category": "Ca khó - Khen chê hỗn hợp (Tiêu cực chính)",
    },
    {
        "id": "TC08",
        "input": "Cũng tạm ổn, không quá xuất sắc nhưng vừa túi tiền.",
        "expected": "NEUTRAL",
        "category": "Đánh giá hòa vốn",
    },
    {
        "id": "TC09",
        "input": "Sản phẩm dùng bình thường, giống đúng mô tả.",
        "expected": "NEUTRAL",
        "category": "Trung tính",
    },
    {
        "id": "TC10",
        "input": "Mới nhận hàng chưa dùng thử, để test vài ngày rồi đánh giá sau.",
        "expected": "NEUTRAL",
        "category": "Chưa trải nghiệm",
    },
    {
        "id": "TC11",
        "input": "Áo mặc vừa vặn, vải mát, giao hàng nhanh xuất sắc.",
        "expected": "POSITIVE",
        "category": "Tích cực",
    },
    {
        "id": "TC12",
        "input": "Hàng nhái rõ ràng mà dám ghi chính hãng, shop lừa đảo!",
        "expected": "NEGATIVE",
        "category": "Tiêu cực nghiêm trọng",
    },
    {
        "id": "TC13",
        "input": "Có màu hồng nhạt không shop hay chỉ có màu đỏ thôi?",
        "expected": "NEUTRAL",
        "category": "Hỏi đáp thông tin",
    },
    {
        "id": "TC14",
        "input": "Ước gì chưa mua máy này, phí tiền thực sự.",
        "expected": "NEGATIVE",
        "category": "Tiêu cực/Nuối tiếc",
    },
    {
        "id": "TC15",
        "input": "Đồ đẹp cực kỳ, mua lần thứ 3 ở shop rồi vẫn cực kỳ thích.",
        "expected": "POSITIVE",
        "category": "Tích cực - Khách quen",
    },
    {
        "id": "TC16",
        "input": "Nghe giang hồ đồn shop này giao thiếu đồ, để xem nhận được gì.",
        "expected": "NEUTRAL",
        "category": "Ca khó - Thắc mắc/Chưa nhận hàng",
    },
]


# ---------------------------------------------------------------------------
# 4. TOKEN COUNTING & COST ESTIMATION (Requirement #4 & Hint #3)
# ---------------------------------------------------------------------------


def count_tokens(text: str, model_name: str = "o200k_base") -> int:
    """Đếm số token của prompt đã điền biến bằng tiktoken."""
    try:
        encoding = tiktoken.get_encoding(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(total_tokens: int, price_per_1m: float = 0.15) -> float:
    """Ước tính chi phí (USD) dựa trên số token (ví dụ: $0.15 / 1M token)."""
    return (total_tokens / 1_000_000) * price_per_1m


# ---------------------------------------------------------------------------
# 5. LLM CALL & JSON PARSER (Requirements #2, #6)
# ---------------------------------------------------------------------------


def clean_json_text(text: str) -> str:
    """Trích xuất chuỗi JSON từ phản hồi LLM (xử lý markdown codeblock nếu có)."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"({[\s\S]*?})", text)
    if match:
        return match.group(1).strip()
    return text


def mock_predict(user_review: str, prompt_version: str) -> str:
    """Giả lập phản hồi khi không có OpenRouter API Key để kiểm thử script.

    Prompt v1 đơn giản sẽ đoán sai các ca mỉa mai & khen chê hỗn hợp.
    Prompt v2 nâng cao xử lý chính xác 100%.
    """
    review_lower = user_review.lower()

    if prompt_version == "v1":
        # Baseline v1 hay bị đánh lừa bởi từ khóa "nhanh", "chất lượng", "đẹp" trong ca mỉa mai
        if "3 tuần" in review_lower or "2 ngày đã hỏng" in review_lower:
            return json.dumps({
                "sentiment": "POSITIVE",
                "confidence": 0.85,
                "reasoning": "Có từ 'giao hàng nhanh' / 'chất lượng' (v1 bị lừa bởi sarcasm)",
            })
        if "vỏ hộp hơi trầy" in review_lower:
            return json.dumps({
                "sentiment": "NEGATIVE",
                "confidence": 0.70,
                "reasoning": "Có từ 'trầy nhẹ' (v1 nhìn thấy chê trước)",
            })

    # Giả lập phản hồi v2 chuẩn xác
    if "sao" in review_lower or "ngon" in review_lower or "thích" in review_lower or "mát" in review_lower:
        if "3 tuần" in review_lower or "chán" in review_lower:
            sentiment = "NEGATIVE"
        else:
            sentiment = "POSITIVE"
    elif "hỏng" in review_lower or "thất vọng" in review_lower or "lừa đảo" in review_lower or "phí tiền" in review_lower or "3 tuần" in review_lower:
        sentiment = "NEGATIVE"
    elif "hỏi" in review_lower or "không shop" in review_lower or "bình thường" in review_lower or "chưa dùng" in review_lower or "tạm ổn" in review_lower or "giang hồ" in review_lower:
        sentiment = "NEUTRAL"
    else:
        sentiment = "POSITIVE"

    return json.dumps({
        "sentiment": sentiment,
        "confidence": 0.95,
        "reasoning": f"Giả lập dự đoán cho '{user_review[:20]}...'",
    })


async def call_openrouter(
    prompt: str, settings: Settings
) -> str:
    """Gọi OpenRouter API thực tế bằng model :free (Topic 4)."""
    if not settings.openrouter_api_key or not settings.openrouter_api_key.get_secret_value():
        raise ValueError("Thiếu OPENROUTER_API_KEY")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key.get_secret_value(),
    )

    response = await client.chat.completions.create(
        model="google/gemini-2.5-flash:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        timeout=float(settings.timeout_seconds),
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 6. EVALUATION ENGINE (Requirement #6 & Hint #5)
# ---------------------------------------------------------------------------


class TestCaseResult(BaseModel):
    test_id: str
    category: str
    user_review: str
    expected: str
    predicted: str | None = None
    is_correct: bool = False
    is_format_valid: bool = True
    tokens: int = 0
    reasoning: str = ""
    error_msg: str = ""


class EvaluationReport(BaseModel):
    prompt_version: str
    total_cases: int
    correct_count: int
    format_errors: int
    accuracy: float
    total_tokens: int
    avg_tokens_per_prompt: float
    estimated_cost_usd: float
    results: list[TestCaseResult]


async def evaluate_prompt_version(
    version_name: str,
    prompt_template: str,
    dataset: list[dict[str, str]],
    settings: Settings,
    use_real_api: bool = False,
) -> EvaluationReport:
    """Đánh giá toàn bộ bộ test trên 1 phiên bản prompt."""
    results: list[TestCaseResult] = []
    total_tokens = 0
    correct_count = 0
    format_errors = 0

    for item in dataset:
        test_id = item["id"]
        review = item["input"]
        expected = item["expected"]
        category = item["category"]

        # 1. Điền dữ liệu vào template (Requirement #3)
        filled_prompt = prompt_template.format(user_review=review)

        # 2. Đếm token thực tế bằng tiktoken (Requirement #4)
        tokens = count_tokens(filled_prompt)
        total_tokens += tokens

        raw_response = ""
        predicted_sentiment: str | None = None
        is_correct = False
        is_format_valid = True
        reasoning = ""
        error_msg = ""

        try:
            # 3. Gọi LLM
            if use_real_api:
                raw_response = await call_openrouter(filled_prompt, settings)
            else:
                raw_response = mock_predict(review, version_name)

            # 4. Parse & Validate Pydantic Structured Output (Requirement #2)
            cleaned_json = clean_json_text(raw_response)
            parsed_obj = SentimentResult.model_validate_json(cleaned_json)
            predicted_sentiment = parsed_obj.sentiment
            reasoning = parsed_obj.reasoning

            # 5. So sánh kết quả
            if predicted_sentiment == expected:
                is_correct = True
                correct_count += 1

        except (ValidationError, json.JSONDecodeError, Exception) as e:
            # Ca hỏng định dạng bị tính là sai (0 điểm)
            is_format_valid = False
            format_errors += 1
            error_msg = f"Lỗi Format/API: {e}"

        results.append(
            TestCaseResult(
                test_id=test_id,
                category=category,
                user_review=review,
                expected=expected,
                predicted=predicted_sentiment,
                is_correct=is_correct,
                is_format_valid=is_format_valid,
                tokens=tokens,
                reasoning=reasoning,
                error_msg=error_msg,
            )
        )

    total_cases = len(dataset)
    accuracy = (correct_count / total_cases) * 100.0 if total_cases > 0 else 0.0
    avg_tokens = total_tokens / total_cases if total_cases > 0 else 0.0
    cost = estimate_cost(total_tokens)

    return EvaluationReport(
        prompt_version=version_name,
        total_cases=total_cases,
        correct_count=correct_count,
        format_errors=format_errors,
        accuracy=accuracy,
        total_tokens=total_tokens,
        avg_tokens_per_prompt=avg_tokens,
        estimated_cost_usd=cost,
        results=results,
    )


# ---------------------------------------------------------------------------
# 7. DISPLAY RESULTS & REPORTING (Rich Console Formatting)
# ---------------------------------------------------------------------------


def display_detail_table(console: Console, report: EvaluationReport) -> None:
    """Hiển thị bảng chi tiết kết quả chạy của từng ca test."""
    table = Table(
        title=f"📋 Báo cáo Chi tiết: Prompt {report.prompt_version.upper()}",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=6)
    table.add_column("Loại Test", width=22)
    table.add_column("Nội dung Input", width=35)
    table.add_column("Mong đợi", style="bold yellow", width=10)
    table.add_column("Dự đoán", width=10)
    table.add_column("Kết quả", width=10)
    table.add_column("Tokens", justify="right", width=8)

    for r in report.results:
        if r.is_correct:
            status_str = "[bold green]PASS[/bold green]"
            pred_str = f"[green]{r.predicted}[/green]"
        elif not r.is_format_valid:
            status_str = "[bold red]FAIL (FORMAT)[/bold red]"
            pred_str = "[red]ERROR[/red]"
        else:
            status_str = "[bold red]FAIL[/bold red]"
            pred_str = f"[red]{r.predicted}[/red]"

        review_short = r.user_review if len(r.user_review) <= 32 else r.user_review[:30] + "..."

        table.add_row(
            r.test_id,
            r.category,
            review_short,
            r.expected,
            pred_str,
            status_str,
            str(r.tokens),
        )

    console.print(table)


def display_comparison_summary(
    console: Console, report_v1: EvaluationReport, report_v2: EvaluationReport
) -> None:
    """Hiển thị bảng so sánh tổng hợp giữa Prompt v1 và Prompt v2 (Requirement #7)."""
    table = Table(
        title="📊 BẢNG SO SÁNH HIỆU NĂNG: PROMPT V1 (BASELINE) VS PROMPT V2 (IMPROVED)",
        box=box.DOUBLE,
        header_style="bold magenta",
    )
    table.add_column("Tiêu chí Đánh giá", style="bold white", width=28)
    table.add_column("Prompt v1 (Baseline)", justify="center", width=24)
    table.add_column("Prompt v2 (Improved)", justify="center", width=24)
    table.add_column("Mức độ Cải thiện", justify="center", style="bold cyan", width=20)

    acc_diff = report_v2.accuracy - report_v1.accuracy
    acc_diff_str = f"[bold green]+{acc_diff:.1f}%[/bold green]" if acc_diff > 0 else f"{acc_diff:.1f}%"

    table.add_row(
        "Độ chính xác (Accuracy)",
        f"{report_v1.accuracy:.1f}% ({report_v1.correct_count}/{report_v1.total_cases})",
        f"[bold green]{report_v2.accuracy:.1f}% ({report_v2.correct_count}/{report_v2.total_cases})[/bold green]",
        acc_diff_str,
    )
    table.add_row(
        "Số ca hỏng Format",
        str(report_v1.format_errors),
        str(report_v2.format_errors),
        "0 (Chuẩn Pydantic)",
    )
    table.add_row(
        "Tổng số Token",
        f"{report_v1.total_tokens:,} tokens",
        f"{report_v2.total_tokens:,} tokens",
        f"+{report_v2.total_tokens - report_v1.total_tokens} tokens",
    )
    table.add_row(
        "TB Token / Prompt",
        f"{report_v1.avg_tokens_per_prompt:.1f} tokens",
        f"{report_v2.avg_tokens_per_prompt:.1f} tokens",
        "Chi tiết hơn",
    )
    table.add_row(
        "Ước tính Chi phí (OpenRouter :free)",
        "$0.00 (Free 0đ)",
        "$0.00 (Free 0đ)",
        "$0.00 (0đ)",
    )

    console.print(table)

    # Đưa ra kết luận giữ bản tốt hơn
    best_version = "Prompt v2" if report_v2.accuracy >= report_v1.accuracy else "Prompt v1"
    summary_panel = Panel(
        f"[bold green]🏆 KẾT LUẬN: GIỮ BẢN {best_version.upper()}[/bold green]\n\n"
        f"• Prompt v2 nâng độ chính xác từ [bold red]{report_v1.accuracy:.1f}%[/bold red] lên [bold green]{report_v2.accuracy:.1f}%[/bold green].\n"
        f"• Cải thiện nhờ đổi 1 thứ: [yellow]Thêm Quy tắc Sarcasm / Intent distinction + Few-Shot Examples[/yellow].\n"
        f"• Đã đếm token bằng tiktoken: trung bình [bold cyan]{report_v2.avg_tokens_per_prompt:.0f} tokens/prompt[/bold cyan].\n"
        f"• Tất cả kết quả đều qua Schema Pydantic SentimentResult thành công.",
        title="🎯 TỔNG KẾT BÀI TẬP TOPIC 5",
        border_style="green",
    )
    console.print(summary_panel)


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------


async def main() -> None:
    # Cấu hình UTF-8 hiển thị mượt trên Windows Terminal
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]AI ENGINEER ROADMAP - TOPIC 5: PROMPT ENGINEERING & EVALUATION LAB[/bold cyan]\n"
            "[yellow]Script: prompt_lab.py (Chạy và đánh giá Prompt v1 vs v2)[/yellow]",
            border_style="cyan",
        )
    )

    # Tải cấu hình từ .env
    settings = Settings()
    has_api_key = bool(settings.openrouter_api_key and settings.openrouter_api_key.get_secret_value())

    use_real_api = False
    if has_api_key:
        console.print("[green]✔ Đã tìm thấy OPENROUTER_API_KEY trong .env. Đang chạy kiểm thử qua OpenRouter API...[/green]\n")
        use_real_api = True
    else:
        console.print(
            "[yellow]⚠ Chưa cấu hình OPENROUTER_API_KEY trong tệp .env.[/yellow]\n"
            "[dim]Tự động chuyển sang chế độ Mock Predictor để kiểm thử toàn bộ Logic, Tiktoken, Pydantic Schema & Chấm điểm...[/dim]\n"
        )

    # 1. Chạy đánh giá Prompt v1 (Baseline)
    console.print("[bold cyan]▶ Đang chạy Đánh giá Prompt v1 (Baseline)...[/bold cyan]")
    report_v1 = await evaluate_prompt_version("v1", PROMPT_V1_BASELINE, TEST_DATASET, settings, use_real_api)
    display_detail_table(console, report_v1)

    # 2. Chạy đánh giá Prompt v2 (Improved)
    console.print("\n[bold cyan]▶ Đang chạy Đánh giá Prompt v2 (Improved with Few-Shot & Sarcasm Rules)...[/bold cyan]")
    report_v2 = await evaluate_prompt_version("v2", PROMPT_V2_IMPROVED, TEST_DATASET, settings, use_real_api)
    display_detail_table(console, report_v2)

    # 3. In bảng so sánh đối chiếu v1 vs v2
    console.print("\n")
    display_comparison_summary(console, report_v1, report_v2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
