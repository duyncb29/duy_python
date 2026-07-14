# LLM App Setup & Environment (Topic 2)

Dự án này là một repo mẫu chuẩn hóa năm 2026 cho các ứng dụng tích hợp LLM. Dự án thể hiện cách thức thiết lập môi trường phát triển nhất quán và an toàn, quản lý các khóa bí mật thông qua tệp cấu hình `.env`, xác thực các biến cấu hình bằng `pydantic-settings`, và đảm bảo chất lượng mã nguồn bằng `Ruff` cùng `Mypy`.

## Các Công Cụ & Thư Viện Sử Dụng

1. **Quản lý dự án & môi trường (`uv`)**: Công cụ quản lý gói tốc độ cực nhanh, thay thế hoàn toàn cho `pip`, `venv`, `poetry`.
2. **Kiểm tra và Định dạng Code (`Ruff`)**: Bộ kiểm tra và định dạng mã nguồn Python siêu tốc, tích hợp đầy đủ các quy tắc chuẩn của Python.
3. **Kiểm tra Kiểu Tĩnh (`Mypy`)**: Giúp phát hiện sớm các lỗi kiểu dữ liệu trong quá trình viết mã nguồn, có cấu hình strict check và tích hợp plugin của Pydantic.
4. **Quản lý Cấu Hình (`pydantic-settings`)**: Tải biến cấu hình từ tệp `.env` với cơ chế xác thực mạnh mẽ (validate loại dữ liệu, bắt buộc khai báo API Key).

## Cấu Trúc Dự Án

```text
llm-app/
  ├── .gitignore         # File bỏ qua các tệp không cần thiết khi git push
  ├── .python-version    # Phiên bản Python sử dụng trong dự án (3.13)
  ├── pyproject.toml     # File cấu hình trung tâm (dependencies, Ruff, mypy)
  ├── uv.lock            # File khóa phiên bản chính xác của dependencies
  ├── .env.example       # Mẫu biến môi trường
  ├── .env               # File cấu hình cục bộ (chứa API Key - không đẩy lên Git)
  ├── main.py            # Mã nguồn chính chạy ứng dụng
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
```

### 3. Khởi chạy ứng dụng

Sử dụng `uv run` để khởi chạy ứng dụng chính một cách an toàn thông qua môi trường ảo vừa được tạo:

```bash
python -m uv run python main.py
```

Ứng dụng sẽ tự động tải các biến cấu hình từ `.env` (API Key sẽ được ẩn tự động dưới dạng `**********` để bảo mật khi in ra log nhờ sử dụng lớp `SecretStr` của Pydantic) và chạy giả lập một yêu cầu gọi LLM.

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
