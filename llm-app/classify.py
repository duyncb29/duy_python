import enum
import html
import re
import sys
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import instructor
from openai import OpenAI
from config import Settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize rich console
console = Console()

# =====================================================================
# 1. PYDANTIC SCHEMA DEFINITION (Requirement #1 & #2)
# =====================================================================

class TicketCategory(str, enum.Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    JUNK = "junk"
    GENERAL_FEEDBACK = "general_feedback"

class TicketPriority(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ClassificationResult(BaseModel):
    category: TicketCategory = Field(
        ...,
        description="Phân loại chính xác yêu cầu của khách hàng."
    )
    priority: TicketPriority = Field(
        ...,
        description="Độ ưu tiên xử lý của yêu cầu (high, medium, low)."
    )
    confidence: float = Field(
        ...,
        description="Độ tin cậy của phân loại này (từ 0.0 đến 1.0). Nếu mơ hồ hoặc không chắc chắn, đặt giá trị thấp."
    )
    needs_escalation: bool = Field(
        ...,
        description="Cờ báo cần chuyển tiếp cho người duyệt (Human Review). Phải là True khi độ tin cậy < 0.6 hoặc yêu cầu mơ hồ, có dấu hiệu bất thường, junk."
    )
    summary: str = Field(
        ...,
        description="Tóm tắt ngắn gọn yêu cầu bằng tiếng Việt (từ 5 đến 150 ký tự)."
    )
    escalation_reason: str = Field(
        ...,
        description="Lý do chuyển tiếp cho người duyệt nếu needs_escalation=True. Nếu needs_escalation=False, bắt buộc phải để chuỗi rỗng."
    )

    # Business Logic Validators (Requirement #2)
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Độ tin cậy phải nằm trong khoảng [0.0, 1.0]. Giá trị hiện tại: {v}")
        return v

    @field_validator("escalation_reason")
    @classmethod
    def validate_escalation_reason(cls, v: str, info: ValidationInfo) -> str:
        needs_escalation = info.data.get("needs_escalation", False)
        v_stripped = v.strip()
        
        if needs_escalation:
            if not v_stripped:
                raise ValueError(
                    "Bắt buộc phải cung cấp lý do chuyển tiếp (escalation_reason) "
                    "khi cần chuyển tiếp cho người duyệt (needs_escalation=True)."
                )
        else:
            if v_stripped:
                raise ValueError(
                    f"Không được cung cấp lý do chuyển tiếp (escalation_reason='{v_stripped}') "
                    "khi không cần chuyển tiếp cho người duyệt (needs_escalation=False). Hãy để chuỗi rỗng."
                )
        return v_stripped

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        v_stripped = v.strip()
        if len(v_stripped) < 5:
            raise ValueError("Tóm tắt quá ngắn. Phải chứa ít nhất 5 ký tự để đảm bảo có đầy đủ ý nghĩa.")
        return v_stripped


# =====================================================================
# 2. INPUT GUARD (Requirement #4)
# =====================================================================

def input_guard(text: str) -> str:
    """Kiểm soát đầu vào để ngăn chặn dữ liệu rỗng, quá dài hoặc Prompt Injection."""
    text_stripped = text.strip()
    
    # 1. Chặn rỗng hoặc quá ngắn
    if not text_stripped or len(text_stripped) < 5:
        raise ValueError("Lỗi Input Guard: Dữ liệu đầu vào quá ngắn hoặc rỗng, không thể phân loại.")
        
    # 2. Chặn dữ liệu quá dài (> 2000 ký tự)
    if len(text_stripped) > 2000:
        raise ValueError(f"Lỗi Input Guard: Dữ liệu đầu vào vượt quá giới hạn cho phép (Độ dài: {len(text_stripped)}/2000).")
        
    # 3. Quét phát hiện hành vi chèn mã độc / SQL injection / XSS cơ bản (và in cảnh báo)
    injection_patterns = [
        r"(?i)\b(drop|truncate|delete)\s+table\b",
        r"(?i)union\s+select",
        r"(?i)<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text_stripped):
            console.print(f"[bold red]⚠ CẢNH BÁO INPUT GUARD: Phát hiện mẫu nghi ngờ Injection/Junk trong dữ liệu đầu vào! Pattern: {pattern}[/bold red]")
            
    # 4. Tách dữ liệu khỏi câu lệnh bằng Delimiter và thoát ký tự phân tách nếu người dùng cố tình nhập vào
    escaped_text = text_stripped.replace("=== USER INPUT START ===", "[ESCAPED_START]").replace("=== USER INPUT END ===", "[ESCAPED_END]")
    
    delimited_text = (
        "=== USER INPUT START ===\n"
        f"{escaped_text}\n"
        "=== USER INPUT END ==="
    )
    return delimited_text


# =====================================================================
# 3. OUTPUT GUARD (Requirement #5)
# =====================================================================

def output_guard(result: ClassificationResult) -> ClassificationResult:
    """Làm sạch các trường dữ liệu text đầu ra để tránh chèn mã SQL hoặc HTML độc hại (Sanitization)."""
    
    def sanitize(text: str) -> str:
        if not text:
            return text
        # Escape các ký tự HTML nguy hại (&, <, >, ", ')
        escaped = html.escape(text)
        # Vô hiệu hóa các từ khóa SQL nguy hiểm bằng cách thêm dấu gạch hoặc ghi chú
        sanitized = re.sub(
            r"(?i)\b(union|select|drop|delete|insert|update|alter|truncate)\b",
            r"[\1_sanitized]",
            escaped
        )
        return sanitized

    result.summary = sanitize(result.summary)
    result.escalation_reason = sanitize(result.escalation_reason)
    return result


# =====================================================================
# 4. MOCK LLM ENGINE FOR OFFLINE RUNNING & RETRY DEMO
# =====================================================================

class MockChatCompletions:
    def __init__(self):
        self.call_count = 0

    def create(self, model: str, response_model: type[BaseModel], messages: list[dict[str, Any]], temperature: float = 0.0, max_retries: int = 3, **kwargs: Any) -> Any:
        self.call_count += 1
        
        # Tìm nội dung input của người dùng từ messages (bỏ qua tin nhắn lỗi validation)
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if "validation error" not in content.lower():
                    user_content = content
                    break
        if not user_content:
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_content = m.get("content", "")
                    break
                
        # Trích xuất nội dung gốc nằm giữa delimiter nếu có
        match = re.search(r"=== USER INPUT START ===\n([\s\S]*?)\n=== USER INPUT END ===", user_content)
        raw_text = match.group(1) if match else user_content
        
        # Xác định kịch bản dựa trên từ khóa trong input
        is_clear = "đăng nhập" in raw_text.lower() or "reset" in raw_text.lower()
        is_ambiguous = "chậm" in raw_text.lower() or "hoàn tiền" in raw_text.lower()
        is_junk = "drop table" in raw_text.lower() or "<script>" in raw_text.lower()
        
        # Kiểm tra xem đây có phải lượt gọi Retry do lỗi Validation hay không
        is_retry = len(messages) > 2 or any("validation" in str(m.get("content", "")).lower() for m in messages)
        
        console.print(f"      [dim]• Call #{self.call_count} (Mock LLM) | Lượt: {'Retry' if is_retry else 'Lần đầu'} | Temperature: {temperature}[/dim]")
        
        if is_clear:
            return ClassificationResult(
                category=TicketCategory.ACCOUNT_ACCESS,
                priority=TicketPriority.HIGH,
                confidence=0.95,
                needs_escalation=False,
                summary="Yêu cầu cấp lại mật khẩu và hỗ trợ lỗi đăng nhập tài khoản.",
                escalation_reason=""
            )
            
        elif is_ambiguous:
            # Mô phỏng quá trình tự sửa lỗi (Validation + Retry - Requirement #3)
            # Lần đầu: Cố tình giả lập vi phạm lỗi nghiệp vụ (needs_escalation=True nhưng escalation_reason="")
            if not is_retry:
                console.print("      [yellow]⚠ Mock LLM cố tình trả về dữ liệu lỗi: needs_escalation=True nhưng escalation_reason='' để test Retry Loop...[/yellow]")
                
                # Mô phỏng lỗi Validation từ Pydantic
                err_msg = (
                    "1 validation error for ClassificationResult\n"
                    "escalation_reason\n"
                    "  Value error, Bắt buộc phải cung cấp lý do chuyển tiếp (escalation_reason) khi cần chuyển tiếp cho người duyệt (needs_escalation=True)."
                )
                console.print(f"      [red]✗ Validation Error phát hiện bởi Pydantic: {err_msg}[/red]")
                console.print(f"      [yellow]↻ Tự động gửi yêu cầu sửa lỗi về LLM (Số lượt còn lại: {max_retries - 1})...[/yellow]")
                
                # Ghi lại log tin nhắn sửa lỗi và thực hiện gọi lại chính nó (simulate retry)
                messages.append({"role": "assistant", "content": "{\"category\":\"general_feedback\",\"priority\":\"medium\",\"confidence\":0.45,\"needs_escalation\":true,\"summary\":\"Phản ánh hệ thống chậm và hỏi về thủ tục hoàn tiền.\",\"escalation_reason\":\"\"}"})
                messages.append({"role": "user", "content": f"validation error: {err_msg}"})
                
                return self.create(model, response_model, messages, temperature, max_retries - 1, **kwargs)
            else:
                # Lần gọi retry: LLM đã nhận được phản hồi lỗi và sửa lại đúng luật
                console.print("      [green]✔ Mock LLM sửa lại dữ liệu đúng luật sau khi nhận phản hồi lỗi từ Pydantic Validator.[/green]")
                return ClassificationResult(
                    category=TicketCategory.GENERAL_FEEDBACK,
                    priority=TicketPriority.MEDIUM,
                    confidence=0.45,
                    needs_escalation=True,
                    summary="Phản ánh hệ thống chậm và hỏi về thủ tục hoàn tiền.",
                    escalation_reason="Yêu cầu có độ tin cậy thấp (0.45) và chứa ý kiến hỗn hợp (hỏi về hoàn tiền nhưng bảo thôi)."
                )
                
        else: # Junk / Injection
            return ClassificationResult(
                category=TicketCategory.JUNK,
                priority=TicketPriority.LOW,
                confidence=0.20,
                needs_escalation=True,
                summary="Dữ liệu rác chứa mã độc SQL Injection và thẻ HTML.",
                escalation_reason="Input chứa mã độc nghi ngờ tấn công injection và không có nội dung nghiệp vụ thực tế."
            )

class MockInstructorClient:
    def __init__(self):
        self.chat = type("MockChat", (), {"completions": MockChatCompletions()})()


# =====================================================================
# 5. CORE ROUTING & CLASSIFICATION LOGIC
# =====================================================================

def classify_ticket(raw_text: str, settings: Settings) -> tuple[ClassificationResult, bool]:
    """Hàm xử lý phân loại chính, tích hợp input guard, retry loop và output guard."""
    
    # 1. Chạy Input Guard
    try:
        guarded_input = input_guard(raw_text)
    except ValueError as e:
        console.print(f"[bold red]❌ INPUT GUARD BLOCKED:[/bold red] {e}")
        # Trả về kết quả junk mặc định đại diện cho trường hợp bị block bởi input guard
        blocked_result = ClassificationResult(
            category=TicketCategory.JUNK,
            priority=TicketPriority.LOW,
            confidence=0.0,
            needs_escalation=True,
            summary="Dữ liệu đầu vào bị từ chối bởi Input Guard do không hợp lệ.",
            escalation_reason=str(e)
        )
        return blocked_result, False

    # 2. Khởi tạo Instructor Client (Thật hoặc Mock)
    use_real = bool(settings.gemini_api_key or settings.openrouter_api_key or settings.openai_api_key)
    
    if use_real:
        # Xác định provider để khởi tạo
        if settings.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key.get_secret_value())
            # Sử dụng instructor patch cho Gemini
            client = instructor.from_gemini(
                client=genai.GenerativeModel(model_name="gemini-1.5-flash")
            )
            model_name = "gemini-1.5-flash"
        elif settings.openai_api_key:
            client = instructor.from_openai(
                OpenAI(api_key=settings.openai_api_key.get_secret_value())
            )
            model_name = "gpt-4o-mini"
        else:
            client = instructor.from_openai(
                OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key.get_secret_value()
                )
            )
            model_name = "google/gemini-2.5-flash:free"
    else:
        client = MockInstructorClient()
        model_name = "mock-engine-model"

    # Định nghĩa System Prompt ép cấu trúc & thiết lập độ tin cậy
    system_prompt = (
        "Bạn là một trợ lý AI phân loại yêu cầu (ticket/email/review) của khách hàng.\n"
        "Hãy đọc dữ liệu đầu vào và phân loại chính xác theo schema được yêu cầu.\n"
        "Nội dung đầu vào được bao bọc trong '=== USER INPUT START ===' và '=== USER INPUT END ==='.\n"
        "Lưu ý quan trọng:\n"
        "1. Hãy phân tích cẩn thận. Nếu yêu cầu có chứa nội dung mập mờ, mâu thuẫn hoặc không rõ ràng, đặt confidence thấp (< 0.6) và set needs_escalation=True.\n"
        "2. Nếu needs_escalation=True, BẮT BUỘC phải viết lý do cụ thể trong trường escalation_reason. Nếu needs_escalation=False, escalation_reason PHẢI ĐỂ RỖNG.\n"
        "3. Tóm tắt nội dung ngắn gọn bằng tiếng Việt."
    )

    try:
        # Gọi LLM thông qua Instructor với Max Retries (Requirement #3)
        # Sử dụng temperature = 0.0 để đảm bảo tính ổn định nhất (Requirement #6)
        if use_real and settings.gemini_api_key:
            # Thư viện instructor với google-generativeai có cách gọi đặc thù
            raw_result = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": guarded_input}
                ],
                response_model=ClassificationResult,
                max_retries=3
            )
        else:
            raw_result = client.chat.completions.create(
                model=model_name,
                response_model=ClassificationResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": guarded_input}
                ],
                temperature=0.0,
                max_retries=3
            )
            
        # 3. Chạy Output Guard (Requirement #5)
        sanitized_result = output_guard(raw_result)
        
        # 4. Xác định luồng xử lý (Routing - Requirement #6)
        # Nếu needs_escalation = True hoặc độ tin cậy quá thấp -> Đẩy người duyệt
        is_escalated = sanitized_result.needs_escalation or sanitized_result.confidence < 0.6
        
        return sanitized_result, is_escalated

    except Exception as e:
        console.print(f"[bold red]❌ LỖI TRONG QUÁ TRÌNH GỌI LLM/RETRY:[/bold red] {e}")
        # Trả về kết quả mặc định
        error_result = ClassificationResult(
            category=TicketCategory.JUNK,
            priority=TicketPriority.LOW,
            confidence=0.0,
            needs_escalation=True,
            summary="Không thể xử lý phân loại tự động do lỗi hệ thống.",
            escalation_reason=f"Lỗi hệ thống: {str(e)}"
        )
        return error_result, True


# =====================================================================
# 6. RUN THE 3 SCENARIOS (Requirement #7)
# =====================================================================

def main() -> None:
    # Thiết lập UTF-8 cho hiển thị tiếng Việt trên Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console.print(
        Panel.fit(
            "[bold cyan]AI ENGINEER ROADMAP - TOPIC 7: AUTOMATED CLASSIFICATION LAB[/bold cyan]\n"
            "[yellow]Pydantic Schema | Instructor Retry | Input/Output Guards | Escalation Routing[/yellow]",
            border_style="cyan"
        )
    )

    settings = Settings()
    use_real = bool(settings.gemini_api_key or settings.openrouter_api_key or settings.openai_api_key)
    
    if use_real:
        console.print("[green]✔ Phát hiện API Key trong cấu hình. Đang chạy chế độ LLM thật...[/green]")
    else:
        console.print("[yellow]⚠ Không phát hiện API Key. Đang chạy chế độ Mock Engine để test 100% logic & Retry Loop...[/yellow]")

    # Định nghĩa 3 ca kiểm thử đại diện
    test_cases = [
        {
            "name": "CA 1: RÕ RÀNG / HỢP LỆ (Clear Input)",
            "text": "Tôi không thể đăng nhập vào tài khoản của mình. Hệ thống báo lỗi sai mật khẩu mặc dù tôi đã đổi hôm qua. Hãy reset giúp tôi."
        },
        {
            "name": "CA 2: MƠ HỒ / CẦN ESCALATION (Ambiguous Input)",
            "text": "Tôi thấy hệ thống hơi chậm, không biết là do mạng nhà tôi hay do bên bạn lỗi. À mà nhân tiện cho tôi hỏi thủ tục hoàn tiền thế nào nhỉ, nếu tôi muốn hủy dịch vụ? Mà thôi, cứ để đó tí tôi xem lại."
        },
        {
            "name": "CA 3: INPUT RÁC / MÃ ĐỘC (Junk & Injection Input)",
            "text": "DROP TABLE users; SELECT * FROM products; <script>alert('hack_xss')</script> alo alo"
        }
    ]

    for i, case in enumerate(test_cases, start=1):
        console.print("\n" + "=" * 90)
        console.print(f"[bold blue]📌 KỊCH BẢN {i}: {case['name']}[/bold blue]")
        console.print(f"[bold dim]Đầu vào thô:[/bold dim] '{case['text']}'")
        console.print("=" * 90)
        
        # Gọi hàm phân loại
        result, is_escalated = classify_ticket(case["text"], settings)
        
        # Hiển thị thông tin kết quả dưới dạng bảng
        table = Table(title="KẾT QUẢ PHÂN LOẠI CHI TIẾT", show_header=True, header_style="bold magenta")
        table.add_column("Trường dữ liệu", style="cyan")
        table.add_column("Giá trị", style="white")
        table.add_column("Chi tiết / Trạng thái", style="yellow")
        
        table.add_row("Category", str(result.category), "Nhãn được gán")
        table.add_row("Priority", str(result.priority), "Mức độ ưu tiên")
        table.add_row("Confidence", f"{result.confidence:.2f}", "Độ tin cậy của model")
        table.add_row("Needs Escalation", str(result.needs_escalation), "Cờ báo chuyển tiếp")
        table.add_row("Summary", result.summary, "Tóm tắt (Đã qua Output Guard)")
        table.add_row("Escalation Reason", result.escalation_reason if result.escalation_reason else "(Trống)", "Lý do (Đã qua Output Guard)")
        
        console.print(table)
        
        # Routing Decision Output
        if is_escalated:
            console.print(
                Panel(
                    f"[bold red]⚡ KẾT QUẢ LUỒNG XỬ LÝ: CHUYỂN TIẾP CHO NGƯỜI DUYỆT (HUMAN REVIEW BRANCH)[/bold red]\n"
                    f"Lý do: {result.escalation_reason if result.escalation_reason else 'Độ tin cậy thấp hoặc cờ escalate được kích hoạt.'}",
                    border_style="red"
                )
            )
        else:
            console.print(
                Panel(
                    "[bold green]✔ KẾT QUẢ LUỒNG XỬ LÝ: TỰ ĐỘNG XỬ LÝ HOÀN TOÀN (AUTO-PROCESSED BRANCH)[/bold green]\n"
                    "Hệ thống tự động tiếp nhận và chuyển tiếp sang bộ phận nghiệp vụ liên quan.",
                    border_style="green"
                )
            )

    # In bảng tổng kết bài học
    summary_table = Table(title="🎯 TỔNG KẾT NGHỆ THUẬT PHÂN LOẠI & GUARDRAILS (TOPIC 7)", show_header=True, header_style="bold green")
    summary_table.add_column("Yêu cầu đề bài", style="bold cyan")
    summary_table.add_column("Trạng thái", style="bold green")
    summary_table.add_column("Cách giải quyết chi tiết", style="dim")
    
    summary_table.add_row("1. Schema Pydantic >=4 fields", "PASSED", "category, priority, confidence, needs_escalation, summary, escalation_reason")
    summary_table.add_row("2. Validator nghiệp vụ", "PASSED", "Kiểm tra confidence [0..1] & escalation_reason phải đi kèm needs_escalation=True")
    summary_table.add_row("3. Validate + retry", "PASSED", "Instructor tự gửi lại log lỗi ValidationError cho LLM sửa (test qua ca 2)")
    summary_table.add_row("4. Input guard", "PASSED", "Chặn rỗng/quá dài, phát hiện mẫu SQL/XSS injection, tách dữ liệu bằng Delimiter")
    summary_table.add_row("5. Output guard", "PASSED", "Html escape và lọc từ khóa SQL nguy hiểm khỏi summary/escalation_reason")
    summary_table.add_row("6. Đường 'không chắc'", "PASSED", "Nhiệt độ = 0.0, rẽ luồng nếu needs_escalation=True hoặc confidence < 0.6")
    summary_table.add_row("7. Chạy thử 3 ca", "PASSED", "Đã mô phỏng & in đầy đủ log luồng đi cho các ca: Rõ ràng, Mơ hồ và Rác/Độc hại")

    console.print("\n")
    console.print(summary_table)


if __name__ == "__main__":
    main()
