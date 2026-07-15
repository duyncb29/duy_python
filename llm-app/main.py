import asyncio
import sys

from pydantic import BaseModel

from config import Settings


class LLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class LLMResponse(BaseModel):
    status: str
    response_text: str
    model_used: str


async def call_llm(request: LLMRequest, settings: Settings) -> LLMResponse:
    """Giả lập gọi LLM sử dụng cấu hình từ BaseSettings."""
    print(f"Khởi chạy yêu cầu gọi LLM model: {settings.llm_model}...")
    print(f"Timeout cấu hình: {settings.timeout_seconds} giây")
    # Giả lập độ trễ mạng
    await asyncio.sleep(1)

    response_text = f"Phản hồi giả lập cho prompt: '{request.prompt}'"
    return LLMResponse(
        status="success",
        response_text=response_text,
        model_used=settings.llm_model,
    )


async def main() -> None:
    # Cấu hình UTF-8 cho terminal để hiển thị tiếng Việt trên Windows không bị lỗi
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("--- Đang tải cấu hình từ .env... ---")
    try:
        settings = Settings()
    except Exception as e:
        print(f"Lỗi tải cấu hình: {e}")
        sys.exit(1)

    print(f"Môi trường: {settings.app_env}")
    # API Key sẽ tự động hiển thị dạng masked (ví dụ: *********) nhờ SecretStr
    print(f"LLM API Key: {settings.llm_api_key}")
    print(f"LLM Model: {settings.llm_model}")

    # Chạy demo
    request = LLMRequest(prompt="Xin chào, đây là câu hỏi test.")
    response = await call_llm(request, settings)

    print("\n--- Kết quả phản hồi từ LLM ---")
    print(f"Trạng thái: {response.status}")
    print(f"Model sử dụng: {response.model_used}")
    print(f"Nội dung: {response.response_text}")


if __name__ == "__main__":
    asyncio.run(main())
