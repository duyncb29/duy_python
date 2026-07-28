import os
import sys
import subprocess
import time

# Đảm bảo hiển thị tiếng Việt trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_script(cmd_list: list[str], cwd: str = ".") -> tuple[bool, str]:
    """Chạy một câu lệnh subprocess và trả về trạng thái cùng output."""
    try:
        # Sử dụng shell=True trên Windows để đảm bảo tương thích tốt các lệnh như 'python -m uv'
        result = subprocess.run(
            cmd_list,
            cwd=cwd,
            text=True,
            capture_output=True,
            encoding="utf-8",
            shell=True,
            timeout=90 # Giới hạn tối đa 1.5 phút cho mỗi bài để tránh treo
        )
        success = result.returncode == 0
        output = result.stdout if success else f"{result.stdout}\n[LỖI CHẠY LỆNH]:\n{result.stderr}"
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Lỗi: Quá thời gian thực thi (Timeout 90s)"
    except Exception as e:
        return False, f"Lỗi hệ thống khi chạy script: {e}"

def main() -> None:
    print("=" * 65)
    print("      TRÌNH CHẠY TỔNG HỢP TOÀN BỘ CÁC BÀI TẬP (TOPIC 1 - 9)     ")
    print("=" * 65)
    print("Hệ thống sẽ chạy tuần tự các bài tập và hiển thị tóm tắt báo cáo.\n")

    # Danh sách các bài tập cần chạy
    # Cấu trúc: (Tên hiển thị, Thư mục chạy, Lệnh chạy, Giải thích)
    topics = [
        (
            "Topic 1: Async Text Pipeline",
            ".",
            ["python", "async_text_pipeline/processor.py"],
            "Xử lý văn bản bất đồng bộ, Cache, Semaphore, Retry."
        ),
        (
            "Topic 2: Cấu hình & Tooling",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "main.py"],
            "Quản lý cấu hình .env qua pydantic-settings, linter Ruff/Mypy."
        ),
        (
            "Topic 3: API Data Collector",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "pipeline.py"],
            "Crawl API song song với httpx, xử lý retry tenacity, lưu out.jsonl."
        ),
        (
            "Topic 4: Model Comparison",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "compare.py", "Hãy viết 1 câu chào ngắn."],
            "So sánh song song phản hồi từ nhiều nhà cung cấp LLM khác nhau."
        ),
        (
            "Topic 5: Prompt Evaluation Lab",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "prompt_lab.py"],
            "Đánh giá Prompt V1/V2, đo lượng token đếm bằng tiktoken."
        ),
        (
            "Topic 6: Tool Calling Assistant",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "assistant.py"],
            "Trợ lý AI tự gọi các hàm Python thực tế (calculate, weather)."
        ),
        (
            "Topic 7: Ticket Classification",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "classify.py"],
            "Phân loại ticket có cấu trúc với instructor, ràng buộc validator."
        ),
        (
            "Topic 8: Mini RAG Scratch",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "my_rag.py"],
            "RAG in-memory tự viết (độ tương đồng Cosine bằng Numpy)."
        ),
        (
            "Topic 9: Production RAG Index",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "index.py"],
            "LangChain & Chroma RAG: Tách nạp PDF tài liệu thực tế và index."
        ),
        (
            "Topic 9: Production RAG Ask Tests",
            "llm-app",
            ["python", "-m", "uv", "run", "python", "ask.py", "--run-tests"],
            "LangChain & Chroma RAG: Kiểm thử 3 câu hỏi, grounding, trích dẫn nguồn."
        ),
    ]

    summary_results = []

    for name, cwd, cmd, desc in topics:
        print(f"\n🚀 Đang chạy: [bold]{name}[/bold]")
        print(f"   Mô tả: {desc}")
        print(f"   Thư mục chạy: {cwd}")
        print(f"   Lệnh thực thi: {' '.join(cmd)}")
        print("-" * 50)
        
        start_time = time.perf_counter()
        success, output = run_script(cmd, cwd=cwd)
        elapsed = time.perf_counter() - start_time
        
        # In tóm tắt kết quả chạy ra màn hình terminal
        if success:
            print("[THÀNH CÔNG] Đã hoàn thành.")
            # In 5 dòng cuối cùng của output để xem tóm tắt kết quả
            lines = output.strip().split("\n")
            preview_lines = lines[-8:] if len(lines) > 8 else lines
            print(">>> KẾT QUẢ ĐẦU RA MẪU:")
            for line in preview_lines:
                print(f"    {line}")
        else:
            print(f"[CẢNH BÁO/LỖI] Bài chạy không thành công hoặc cảnh báo cấu hình.")
            # In 10 dòng đầu của lỗi để xem lý do
            lines = output.strip().split("\n")
            preview_lines = lines[:10]
            print(">>> CHI TIẾT LỖI/CẢNH BÁO:")
            for line in preview_lines:
                print(f"    {line}")
                
        summary_results.append({
            "name": name,
            "success": success,
            "time": elapsed
        })
        print("=" * 50)
        time.sleep(1) # Nghỉ 1 giây giữa các bài để hiển thị rõ ràng

    # Hiển thị bảng tổng kết cuối cùng
    print("\n" + "=" * 65)
    print("                  BẢNG TỔNG KẾT KẾT QUẢ CHẠY                   ")
    print("=" * 65)
    print(f"{'Tên Bài Tập (Topic)':<35} | {'Trạng Thái':<12} | {'Thời Gian chạy':<10}")
    print("-" * 65)
    for res in summary_results:
        status_str = "THÀNH CÔNG" if res["success"] else "THẤT BẠI/MOCK"
        print(f"{res['name']:<35} | {status_str:<12} | {res['time']:.2f} giây")
    print("=" * 65)
    print("\n[Gợi ý] Trạng thái 'THẤT BẠI/MOCK' thường xảy ra đối với các bài LLM thực tế (Topic 4) nếu bạn chưa cấu hình API Key thật trong file .env.")
    print("Trình chạy tổng hợp hoàn tất!")

if __name__ == "__main__":
    main()
