# LLM App & API Collector Workspace (Topic 2 & 3 Merged)

Dự án này là một repo mẫu chuẩn hóa năm 2026 tích hợp cả hai bài tập:
1. **Topic 2 - Tooling & Môi trường (LLM App)**: Thiết lập cấu hình hệ thống, quản lý môi trường ảo với `uv` và linter/formatter với `Ruff` + `Mypy`.
2. **Topic 3 - Trình thu thập dữ liệu API (API Collector)**: Xây dựng một pipeline thu thập dữ liệu bất đồng bộ chạy song song, chịu lỗi và validate dữ liệu mạnh mẽ.

Hai phần được kết nối chặt chẽ thông qua hệ thống cấu hình tập trung trong `config.py` sử dụng `pydantic-settings` đọc trực tiếp từ `.env`.

## Các Công Cụ & Thư Viện Sử Dụng

1. **Quản lý dự án & môi trường (`uv`)**: Công cụ quản lý gói tốc độ cực nhanh, thay thế hoàn toàn cho `pip`, `venv`, `poetry`.
2. **Kiểm tra và Định dạng Code (`Ruff`)**: Bộ kiểm tra và định dạng mã nguồn Python siêu tốc, tích hợp đầy đủ các quy tắc chuẩn của Python.
3. **Kiểm tra Kiểu Tĩnh (`Mypy`)**: Giúp phát hiện sớm các lỗi kiểu dữ liệu trong quá trình viết mã nguồn, có cấu hình strict check và tích hợp plugin của Pydantic.
4. **Xác thực Cấu hình (`pydantic-settings`)**: Tải biến cấu hình từ tệp `.env` với cơ chế xác thực mạnh mẽ (validate loại dữ liệu, ẩn API Key bằng `SecretStr`).
5. **HTTP Client (`httpx`)**: Thư viện HTTP client bất đồng bộ mạnh mẽ và hiệu năng cao.
6. **Thử lại tự động (`tenacity`)**: Cung cấp cơ chế retry thông minh với exponential backoff và jitter khi gặp lỗi mạng tạm thời.

## Cấu Trúc Dự Án

```text
llm-app/
  ├── .gitignore         # File bỏ qua các tệp không cần thiết khi git push
  ├── .python-version    # Phiên bản Python sử dụng trong dự án (3.13)
  ├── pyproject.toml     # File cấu hình trung tâm (dependencies, Ruff, mypy)
  ├── uv.lock            # File khóa phiên bản chính xác của dependencies
  ├── .env.example       # Mẫu biến môi trường
  ├── .env               # File cấu hình cục bộ (chứa API Key - không đẩy lên Git)
  ├── config.py          # Lớp cấu hình dùng chung (Settings) sử dụng pydantic-settings
  ├── main.py            # Chạy giả lập gọi LLM (Topic 2)
  ├── pipeline.py        # Chạy pipeline thu thập dữ liệu API (Topic 3)
  ├── out.jsonl          # Tệp tin đầu ra chứa dữ liệu sạch đã được thu thập (JSON Lines)
  └── README.md          # Hướng dẫn này
```

## Hướng Dẫn Thiết Lập & Chạy Ứng Dụng

### 1. Đồng bộ môi trường và cài đặt dependencies

Để thiết lập môi trường ảo `.venv` và tự động cài đặt tất cả thư viện cần thiết, bạn chỉ cần di chuyển vào thư mục dự án và chạy:

```bash
# Di chuyển vào thư mục dự án
cd llm-app

# Thực hiện đồng bộ qua uv
python -m uv sync
```

Lệnh trên sẽ tự động đọc cấu hình từ `pyproject.toml` và `uv.lock`, tạo một môi trường ảo cục bộ `.venv` tách biệt và cực kỳ sạch sẽ.

### 2. Cấu hình biến môi trường

Sao chép file `.env.example` thành file `.env` và tùy chỉnh các tham số cấu hình:

```bash
cp .env.example .env
```

Nội dung `.env` mẫu:
```ini
APP_ENV=development
LLM_API_KEY=sk-mock-api-key-from-env-file-12345
LLM_MODEL=gpt-4o
TIMEOUT_SECONDS=10

# Cấu hình API Collector (Topic 3)
COLLECTOR_BASE_URL=https://jsonplaceholder.typicode.com
COLLECTOR_SEMAPHORE_LIMIT=10
COLLECTOR_TIMEOUT_SECONDS=10
```

### 3. Chạy ứng dụng gọi LLM giả lập (Topic 2)

Sử dụng `uv run` để chạy file `main.py`:

```bash
python -m uv run python main.py
```

### 4. Chạy trình thu thập dữ liệu API (Topic 3)

Sử dụng `uv run` để chạy file `pipeline.py`:

```bash
python -m uv run python pipeline.py
```

Khi chạy, chương trình sẽ tự động nạp cấu hình API, giới hạn kết nối Semaphore và Timeout từ `.env` để tải song song 100 posts (trong đó có 5 ID lỗi cố ý), sau đó ghi 95 bản ghi sạch thành công vào `out.jsonl`.

## Công Cụ Kiểm Tra Chất Lượng Mã Nguồn

### 1. Kiểm tra Linter & Formatter (Ruff)

*   Để kiểm tra mã nguồn (lint):
    ```bash
    python -m uv run ruff check .
    ```
*   Để tự động sửa các lỗi định dạng và thứ tự import:
    ```bash
    python -m uv run ruff check --fix .
    ```
*   Để chạy định dạng mã nguồn (format check):
    ```bash
    python -m uv run ruff format --check .
    ```

### 2. Kiểm tra Kiểu Tĩnh (Mypy)

Mypy đã được cấu hình ở chế độ kiểm tra nghiêm ngặt (`strict = true`) kết hợp cùng plugin `pydantic.mypy` giúp phân tích kiểu cực kỳ chính xác:

```bash
python -m uv run mypy .
```
