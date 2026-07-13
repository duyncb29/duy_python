# Hệ thống Xử lý Văn bản Bất đồng bộ (Async Text Pipeline)

Dự án này triển khai một chương trình xử lý và tóm tắt văn bản bằng Python sử dụng lập trình không đồng bộ (`asyncio`). Hệ thống được thiết kế để đọc các câu hỏi đầu vào từ tệp dữ liệu, xử lý chúng đồng thời và áp dụng các cơ chế kiểm soát lỗi cũng như tối ưu hóa tốc độ.

## Cấu trúc thư mục dự án

```text
async_text_pipeline/
  ├── input_texts.txt    # Tệp dữ liệu chứa danh sách câu hỏi cần xử lý
  └── processor.py       # File mã nguồn chính xử lý bất đồng bộ
.gitignore               # Các tệp loại trừ khỏi Git
README.md                # Tài liệu hướng dẫn sử dụng này
```

## Các tính năng lập trình đã triển khai

1. **Bài 1: Định nghĩa Data Model (Pydantic)**
   - Sử dụng thư viện `pydantic` (`BaseModel`) để tạo lớp dữ liệu `Summary` nhằm xác thực định dạng dữ liệu đầu ra có tính nhất quán (gồm `word_count` kiểu số nguyên và `text` kiểu chuỗi ký tự).

2. **Bài 2: Đọc file dữ liệu & Hàm tóm tắt kèm Cache**
   - Đọc danh sách câu hỏi bất đồng bộ từ file `input_texts.txt` thông qua `asyncio.to_thread` để tránh gây nghẽn (blocking) tiến trình chính.
   - Hàm tóm tắt `summarize` được tích hợp bộ nhớ đệm `_cache` (in-memory cache) giúp nhận diện và trả về ngay kết quả nếu văn bản đầu vào đã từng được xử lý, tăng tốc tối đa tốc độ xử lý.

3. **Bài 3: Đo thời gian thực hiện & Decorator Retry tự động**
   - Sử dụng Context Manager `@contextlib.contextmanager` đặt tên là `measure_time` để đo chính xác thời gian thực thi của các khối mã lệnh.
   - Thiết kế Decorator `@retry` để tự động chạy lại các tác vụ bất đồng bộ nếu xảy ra lỗi kết nối hoặc ngoại lệ trong quá trình chạy (thử lại tối đa 3 lần).

4. **Bài 4: Giới hạn tần suất xử lý (Semaphore) & Hàm điều phối chính**
   - Sử dụng `asyncio.Semaphore(3)` giới hạn tối đa chỉ cho phép 3 tác vụ chạy đồng thời tại một thời điểm để tránh quá tải hệ thống.
   - Hàm `call_llm` sử dụng `asyncio.gather` để kích hoạt đồng thời toàn bộ tác vụ xử lý lô câu hỏi dưới sự kiểm soát của Semaphore.

## Hướng dẫn cài đặt và chạy chương trình

### Yêu cầu hệ thống
- Python 3.9 trở lên
- Thư viện `pydantic`

### Hướng dẫn chạy chương trình

1. Cài đặt thư viện Pydantic:
   ```bash
   pip install pydantic
   ```

2. Chạy chương trình từ thư mục gốc của dự án:
   ```bash
   python async_text_pipeline/processor.py
   ```
