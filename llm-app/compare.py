import asyncio
import sys

from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import Settings


class MissingApiKeyError(Exception):
    """Ngoại lệ khi không tìm thấy API Key cấu hình cho một provider."""

    pass


class ProviderResult(BaseModel):
    """Model biểu diễn kết quả phản hồi của từng nhà cung cấp LLM."""

    provider: str
    success: bool
    status_message: str
    response_text: str = ""


async def ask(provider: str, prompt: str, settings: Settings) -> str:
    """Mặt tiền chung (Unified Interface) gọi API của các LLM SDK khác nhau.

    Đảm bảo truy cập đúng các thuộc tính chứa text phản hồi của từng SDK.
    """
    provider_lower = provider.lower()

    if provider_lower == "openrouter":
        if not settings.openrouter_api_key:
            raise MissingApiKeyError("Thiếu OPENROUTER_API_KEY trong tệp .env")

        # OpenRouter sử dụng thư viện openai tương thích
        from openai import AsyncOpenAI

        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key.get_secret_value(),
        )
        # Sử dụng model miễn phí :free để đáp ứng yêu cầu
        openrouter_response = await openrouter_client.chat.completions.create(
            model="google/gemini-2.5-flash:free",
            messages=[{"role": "user", "content": prompt}],
            timeout=float(settings.timeout_seconds),
        )
        # Lấy text bằng thuộc tính `.content` của OpenAI SDK
        return openrouter_response.choices[0].message.content or ""

    elif provider_lower == "openai":
        if not settings.openai_api_key:
            raise MissingApiKeyError("Thiếu OPENAI_API_KEY trong tệp .env")

        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        openai_response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Sử dụng model tối ưu chi phí để so sánh
            messages=[{"role": "user", "content": prompt}],
            timeout=float(settings.timeout_seconds),
        )
        # Lấy text bằng thuộc tính `.content` của OpenAI SDK
        return openai_response.choices[0].message.content or ""

    elif provider_lower == "gemini":
        if not settings.gemini_api_key:
            raise MissingApiKeyError("Thiếu GEMINI_API_KEY trong tệp .env")

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key.get_secret_value())  # type: ignore[attr-defined]
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore[attr-defined]
        # Gọi bất đồng bộ sử dụng phương thức của Google Generative AI SDK
        gemini_response = await gemini_model.generate_content_async(prompt)
        # Lấy text bằng thuộc tính `.text` của Gemini SDK
        return gemini_response.text or ""

    elif provider_lower == "claude":
        if not settings.anthropic_api_key:
            raise MissingApiKeyError("Thiếu ANTHROPIC_API_KEY trong tệp .env")

        from anthropic import AsyncAnthropic

        claude_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        claude_response = await claude_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            timeout=float(settings.timeout_seconds),
        )
        # Lấy text bằng thuộc tính `.text` của Anthropic SDK (response.content[0].text)
        return claude_response.content[0].text or ""  # type: ignore[union-attr]

    else:
        raise ValueError(f"Provider '{provider}' không được hỗ trợ.")


async def ask_provider_wrapper(
    provider: str, prompt: str, settings: Settings
) -> ProviderResult:
    """Gọi hàm ask() và xử lý lỗi/thiếu key một cách an toàn."""
    try:
        response_text = await ask(provider, prompt, settings)
        return ProviderResult(
            provider=provider,
            success=True,
            status_message="Thành công",
            response_text=response_text.strip(),
        )
    except MissingApiKeyError as e:
        return ProviderResult(
            provider=provider,
            success=False,
            status_message=str(e),
        )
    except Exception as e:
        return ProviderResult(
            provider=provider,
            success=False,
            status_message=f"Lỗi: {e}",
        )


async def main() -> None:
    # Đảm bảo terminal hiển thị đúng tiếng Việt trên Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console()

    # Tải cấu hình qua Settings
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[bold red]Lỗi khi tải cấu hình Settings:[/bold red] {e}")
        sys.exit(1)

    # Đọc prompt từ CLI argument hoặc nhập từ terminal
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        console.print(
            Panel(
                "[bold yellow]Gợi ý:[/bold yellow] Bạn có thể chạy "
                "`python compare.py <prompt>` để so sánh trực tiếp từ CLI.",
                title="Model Comparison Tool",
                box=box.ROUNDED,
            )
        )
        prompt = input("Nhập prompt cần so sánh: ").strip()

    if not prompt:
        console.print("[bold red]Prompt không được để trống![/bold red]")
        sys.exit(1)

    providers = ["openrouter", "openai", "gemini", "claude"]

    console.print(
        f"\n[cyan]Đang gửi prompt tới {len(providers)} nhà cung cấp...[/cyan]"
    )
    console.print(f"[dim]Prompt: '{prompt}'[/dim]\n")

    # Gọi song song các provider
    tasks = [ask_provider_wrapper(p, prompt, settings) for p in providers]
    results: list[ProviderResult] = await asyncio.gather(*tasks)

    # In thông báo các nhà bị bỏ qua do thiếu key hoặc lỗi
    has_errors = False
    for res in results:
        if not res.success:
            has_errors = True
            console.print(
                f"[yellow][!] Bỏ qua / Lỗi [{res.provider.upper()}]: "
                f"{res.status_message}[/yellow]"
            )

    if has_errors:
        console.print()

    # Chuẩn bị bảng so sánh kết quả cạnh nhau
    table = Table(
        title=f"So sánh kết quả cho prompt: '{prompt}'",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        expand=True,
    )

    # Thêm cột cho từng nhà cung cấp
    for res in results:
        title = (
            f"[bold green]{res.provider.upper()}[/bold green]"
            if res.success
            else f"[bold red]{res.provider.upper()} (Lỗi/Thiếu Key)[/bold red]"
        )
        table.add_column(title, ratio=1)

    # Thêm hàng chứa câu trả lời
    row_cells = []
    for res in results:
        if res.success:
            row_cells.append(res.response_text)
        else:
            row_cells.append(f"[dim yellow]{res.status_message}[/dim yellow]")

    table.add_row(*row_cells)

    # Renders bảng so sánh
    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
