import asyncio
import sys
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import Settings


class Post(BaseModel):
    """Pydantic Model biểu diễn thông tin một bài viết."""

    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    id: int
    title: str
    body: str


# Định nghĩa cơ chế retry tự động cho lỗi mạng tạm thời hoặc timeout
retry_on_network_error = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=5),
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    reraise=True,
)


@retry_on_network_error
async def fetch_post_raw(client: httpx.AsyncClient, post_id: int) -> Any:
    """Tải dữ liệu bài viết thô từ API với cơ chế tự động thử lại."""
    response = await client.get(f"/posts/{post_id}")
    # Đảm bảo raise ngoại lệ nếu gặp mã lỗi 4xx/5xx để xử lý chịu lỗi
    response.raise_for_status()
    return response.json()


async def fetch_and_validate_post(
    client: httpx.AsyncClient, post_id: int, semaphore: asyncio.Semaphore
) -> Post:
    """Tải bài viết và validate bằng Pydantic dưới sự kiểm soát của Semaphore."""
    async with semaphore:
        raw_data = await fetch_post_raw(client, post_id)
        # Tiến hành validate dữ liệu bằng Pydantic
        post = Post.model_validate(raw_data)
        return post


async def main() -> None:
    # Cấu hình UTF-8 cho terminal để hiển thị tiếng Việt trên Windows không bị lỗi
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Tải cấu hình dùng chung của dự án qua Pydantic Settings
    print("--- Đang tải cấu hình từ .env... ---")
    try:
        settings = Settings()
    except Exception as e:
        print(f"Lỗi tải cấu hình: {e}")
        sys.exit(1)

    # Danh sách 100 bài viết cần tải
    # (gồm 95 bài viết hợp lệ và 5 ID lỗi để kiểm thử chịu lỗi)
    post_ids = list(range(1, 96)) + [999, 1000, 1001, -5, 0]

    # Cấu hình giới hạn kết nối và timeout lấy từ Settings
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)
    timeout = httpx.Timeout(
        float(settings.collector_timeout_seconds),
        connect=5.0,
    )

    # Khởi tạo Semaphore từ biến cấu hình
    semaphore = asyncio.Semaphore(settings.collector_semaphore_limit)

    print("\n--- Bắt đầu thu thập dữ liệu bất đồng bộ từ API... ---")
    print(f"URL API đích: {settings.collector_base_url}")
    print(f"Tổng số ID bài viết cần tải: {len(post_ids)}")
    print(
        f"Đang xử lý song song với Semaphore (giới hạn "
        f"{settings.collector_semaphore_limit} request đồng thời)..."
    )

    # Sử dụng AsyncClient dùng chung được cấu hình từ Settings
    async with httpx.AsyncClient(
        base_url=settings.collector_base_url,
        timeout=timeout,
        limits=limits,
    ) as client:
        # Chuẩn bị danh sách task bất đồng bộ
        tasks = [fetch_and_validate_post(client, pid, semaphore) for pid in post_ids]

        # Thực thi song song và đón nhận các ngoại lệ
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Phân nhóm kết quả thành công và lỗi
    ok_posts: list[Post] = []
    errors: list[Exception] = []

    for item in results:
        if isinstance(item, Exception):
            errors.append(item)
        elif isinstance(item, Post):
            ok_posts.append(item)
        else:
            errors.append(ValueError(f"Dữ liệu không đúng định dạng Post: {item}"))

    # In thống kê kết quả
    print("\n--- Kết quả thống kê thu thập ---")
    print(f"Thành công (OK): {len(ok_posts)}")
    print(f"Thất bại (Lỗi):  {len(errors)}")
    print(f"Định dạng in:    OK: {len(ok_posts)} / Lỗi: {len(errors)}")

    # Ghi dữ liệu sạch ra file out.jsonl
    output_filename = "out.jsonl"
    print(
        f"\nĐang ghi {len(ok_posts)} bài viết thành công vào tệp {output_filename}..."
    )
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for post in ok_posts:
                # Ghi dưới định dạng JSON Lines
                f.write(post.model_dump_json() + "\n")
        print("Đã ghi file out.jsonl thành công!")
    except Exception as e:
        print(f"Lỗi khi ghi tệp tin: {e}")


if __name__ == "__main__":
    asyncio.run(main())
