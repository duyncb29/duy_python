import asyncio
import functools
import time
import contextlib
import sys
from pathlib import Path
from pydantic import BaseModel

# =====================================================================
# BÀI 1: ĐỊNH NGHĨA DATA MODEL BẰNG PYDANTIC
# =====================================================================
class Summary(BaseModel):
    word_count: int
    text: str

_cache: dict[str, Summary] = {}


# =====================================================================
# BÀI 2: ĐỌC DỮ LIỆU ĐẦU VÀO VÀ HÀM TÓM TẮT (SUMMARIZE) KÈM CACHE
# =====================================================================
async def read_input_file(filepath: Path) -> list[str]:
    """Đọc dữ liệu từ file đầu vào một cách bất đồng bộ."""
    def read_sync():
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return await asyncio.to_thread(read_sync)

async def summarize(text: str) -> Summary:
    """Tóm tắt văn bản với độ trễ giả lập và bộ nhớ đệm (Cache)."""
    if text in _cache:
        return _cache[text]
    
    await asyncio.sleep(1)
    word_count = len(text)
    content = "Summary: " + text
    print(f"Summarizing: word_count={word_count}")
    
    summary = Summary(word_count=word_count, text=content)
    _cache[text] = summary
    return summary


# =====================================================================
# BÀI 3: ĐO THỜI GIAN CHẠY VÀ CƠ CHẾ TỰ ĐỘNG THỬ LẠI (RETRY DECORATOR)
# =====================================================================
@contextlib.contextmanager
def measure_time():
    """Hàm đo thời gian thực thi (Context Manager)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"Thời gian thực hiện: {end - start:.4f} giây")

def retry(times: int):
    """Decorator tự động thử lại khi hàm xảy ra lỗi."""
    def decorator(func):
        @functools.wraps(func)
        async def inner(*args, **kwargs):
            for attempt in range(times):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Lần thử {attempt + 1} thất bại: {e}")
                    if attempt + 1 == times:
                        raise
                    await asyncio.sleep(1)
        return inner
    return decorator


# =====================================================================
# BÀI 4: KIỂM SOÁT TẦN SUẤT GỌI (SEMAPHORE) VÀ HÀM CHẠY CHÍNH (MAIN)
# =====================================================================
semaphore = asyncio.Semaphore(3)

async def rate_limited_summarize(doc_content: str) -> Summary:
    """Gọi hàm tóm tắt văn bản thông qua Semaphore."""
    async with semaphore:
        return await summarize(doc_content)

@retry(times=3)
async def call_llm(doc: list[str]) -> list[Summary]:
    """Xử lý đồng thời cả danh sách văn bản."""
    return await asyncio.gather(*[rate_limited_summarize(d) for d in doc])

async def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / "input_texts.txt"
    
    print(f"Đọc tệp dữ liệu từ: {input_path}")
    if not input_path.exists():
        print(f"Lỗi: Tệp {input_path} không tồn tại.")
        return
        
    texts = await read_input_file(input_path)
    print(f"Đã tải {len(texts)} văn bản cần tóm tắt.\n")
    
    print("--- Chạy thử nghiệm 3 tác vụ đồng thời với Semaphore ---")
    task1 = asyncio.create_task(rate_limited_summarize(texts[0]))
    task2 = asyncio.create_task(rate_limited_summarize(texts[1]))
    task3 = asyncio.create_task(rate_limited_summarize(texts[2]))
    
    t = time.perf_counter()
    await task1
    await task2
    await task3
    print(f"Thời gian chạy 3 tác vụ: {time.perf_counter() - t:.4f} giây\n")
    
    print("--- Bắt đầu xử lý toàn bộ danh sách (call_llm) ---")
    with measure_time():
        results = await call_llm(texts)
        
    print("\n--- Kết quả sau cùng ---")
    for i, summary in enumerate(results):
        print(f"Kết quả {i+1} (Độ dài: {summary.word_count} ký tự): {summary.text}")

if __name__ == "__main__":
    # Cấu hình UTF-8 cho terminal để hiển thị tiếng Việt trên Windows không bị lỗi
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
