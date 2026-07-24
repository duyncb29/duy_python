import glob
import hashlib
import json
import os
import sys

from langchain_core.embeddings import Embeddings

from config import Settings

# Đảm bảo hiển thị tiếng Việt chính xác trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class MockEmbeddings(Embeddings):
    """Mô hình tạo vector embedding giả lập (1536 dims) có tính chất ĐƠN TRỊ (Deterministic) để chạy offline."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import numpy as np
        words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
        vector = np.zeros(1536, dtype=np.float32)
        for w in words:
            if len(w) > 1:
                # Sử dụng md5 để sinh hash mã số cố định, không thay đổi giữa các tiến trình chạy
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % 1536
                vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

def generate_sample_pdf(pdf_path: str) -> None:
    """Tạo tệp PDF tài liệu mẫu để kiểm thử RAG (sử dụng thư viện reportlab)."""
    print(f"--- Đang tạo tài liệu PDF mẫu tại: {pdf_path} ---")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    try:
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
        from reportlab.pdfbase import pdfmetrics  # type: ignore
        from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # type: ignore
    except ImportError:
        print("Lỗi: Không tìm thấy thư viện 'reportlab'.")
        return

    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    arial_path = "C:\\Windows\\Fonts\\arial.ttf"
    arial_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"
    if os.path.exists(arial_path):
        try:
            pdfmetrics.registerFont(TTFont("Arial", arial_path))
            if os.path.exists(arial_bold_path):
                pdfmetrics.registerFont(TTFont("Arial-Bold", arial_bold_path))
                font_bold = "Arial-Bold"
            else:
                font_bold = "Arial"
            font_name = "Arial"
            print("Đã đăng ký font chữ Arial thành công để hiển thị tiếng Việt đẹp mắt.")
        except Exception as e:
            print(f"Cảnh báo: Không thể đăng ký font Arial ({e}). Sử dụng Helvetica mặc định.")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor='#1A365D',
        spaceAfter=15,
        alignment=1
    )

    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=13,
        leading=16,
        textColor='#2C5282',
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor='#2D3748',
        spaceAfter=8
    )

    story = []

    # TRANG 1
    story.append(Paragraph("TÀI LIỆU HƯỚNG DẪN SỬ DỤNG THIẾT BỊ SMARTCHEF X", title_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("1. Giới thiệu chung về SmartChef X", heading_style))
    story.append(Paragraph(
        "Thiết bị SmartChef X là nồi đa năng thông minh tích hợp hơn 15 chế độ nấu tự động, màn hình cảm ứng OLED sắc nét và hỗ trợ kết nối Wifi băng tần 2.4GHz để điều khiển từ xa.",
        body_style
    ))

    story.append(Paragraph("2. Hướng dẫn kết nối Wifi cho SmartChef X", heading_style))
    story.append(Paragraph(
        "Để kết nối SmartChef X với mạng Wifi, trước hết hãy nhấn giữ nút Wifi trên bảng điều khiển trong 5 giây cho đến khi đèn báo nháy nhanh. Tiếp theo, mở ứng dụng SmartLife trên điện thoại, chọn Thêm thiết bị và làm theo hướng dẫn kết nối trên ứng dụng.",
        body_style
    ))

    story.append(Paragraph("3. Chế độ nấu áp suất an toàn", heading_style))
    story.append(Paragraph(
        "Khi sử dụng chế độ nấu áp suất của SmartChef X, van xả áp phải luôn ở vị trí đóng (Sealing). Tuyệt đối không cố gắng mở nắp nồi khi cột chỉ thị áp suất màu đỏ vẫn đang nổi lên. Hãy đợi nồi tự hạ áp suất hoặc nhấn nút xả áp thủ công trước khi mở.",
        body_style
    ))

    story.append(Paragraph("4. Vệ sinh lòng nồi và khay nước ngưng tụ", heading_style))
    story.append(Paragraph(
        "Lòng nồi của SmartChef X được phủ lớp chống dính gốm cao cấp. Hãy vệ sinh lòng nồi bằng nước ấm, xà phòng dịu nhẹ và bọt biển mềm. Không dùng búi sắt hoặc chất tẩy rửa mạnh. Khay chứa nước ngưng tụ ở mặt sau cần tháo và đổ nước sau mỗi lần nấu.",
        body_style
    ))

    story.append(PageBreak())

    # TRANG 2
    story.append(Paragraph("TÀI LIỆU HƯỚNG DẪN SỬ DỤNG THIẾT BỊ SMARTCHEF X - PHẦN 2", title_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("5. Chính sách bảo hành chính hãng", heading_style))
    story.append(Paragraph(
        "Thiết bị SmartChef X được bảo hành chính hãng 24 tháng đối với các lỗi phần cứng phát sinh từ phía nhà sản xuất (như hỏng bảng điều khiển, lỗi cảm biến nhiệt). Các phụ kiện đi kèm như muỗng, xửng hấp được bảo hành 12 tháng.",
        body_style
    ))

    story.append(Paragraph("6. Chính sách đổi trả sản phẩm", heading_style))
    story.append(Paragraph(
        "Khách hàng được quyền đổi mới sản phẩm SmartChef X miễn phí trong vòng 7 ngày đầu kể từ ngày mua nếu sản phẩm gặp lỗi phần cứng kỹ thuật được xác nhận bởi trung tâm bảo hành. Sản phẩm đổi trả phải đầy đủ hộp và phụ kiện đi kèm.",
        body_style
    ))

    story.append(Paragraph("7. Mã lỗi E1 và cách khắc phục", heading_style))
    story.append(Paragraph(
        "Lỗi E1 hiển thị trên màn hình SmartChef X cảnh báo tình trạng nồi bị quá nhiệt (nhiệt độ lòng nồi vượt mức 200 độ C do thiếu nước hoặc bị cháy khét đáy). Cách xử lý: Rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, thêm nước trước khi nấu tiếp.",
        body_style
    ))

    story.append(Paragraph("8. Chế độ nấu chậm (Slow Cook)", heading_style))
    story.append(Paragraph(
        "Chế độ Slow Cook của SmartChef X duy trì nhiệt độ ổn định ở mức 85-90 độ C trong thời gian dài (từ 2 đến 8 giờ tùy cài đặt). Chế độ này lý tưởng cho các món hầm xương, kho cá giúp giữ trọn vẹn hương vị và dưỡng chất.",
        body_style
    ))

    story.append(Paragraph("9. Tải thêm công thức nấu ăn mới", heading_style))
    story.append(Paragraph(
        "Bạn có thể tải thêm hàng trăm công thức nấu ăn miễn phí thông qua kho công thức trực tuyến trên ứng dụng SmartLife. Các công thức mới được cập nhật tự động định kỳ vào ngày 1 hàng tháng.",
        body_style
    ))

    story.append(Paragraph("10. Thông tin liên hệ hỗ trợ kỹ thuật", heading_style))
    story.append(Paragraph(
        "Mọi thắc mắc kỹ thuật về SmartChef X xin vui lòng liên hệ tổng đài chăm sóc khách hàng 1900-8198 (hoạt động từ 8h00 đến 21h00 tất cả các ngày trong tuần) hoặc gửi email trực tiếp tới support@smartchef.vn.",
        body_style
    ))

    doc.build(story)
    print("--- Đã tạo thành công tài liệu PDF mẫu! ---")

def main() -> None:
    print("=== QUY TRÌNH LOAD - SPLIT - EMBED - STORE (INDEX.PY) ===")

    try:
        settings = Settings()
    except Exception as e:
        print(f"Lỗi tải cấu hình: {e}")
        sys.exit(1)

    openai_key = None
    if settings.openai_api_key is not None:
        openai_key = settings.openai_api_key.get_secret_value()

    gemini_key = None
    if settings.gemini_api_key is not None:
        gemini_key = settings.gemini_api_key.get_secret_value()

    if not openai_key and settings.llm_api_key is not None:
        val = settings.llm_api_key.get_secret_value()
        if not val.startswith("sk-mock"):
            openai_key = val

    # Xác định Embedding Provider
    provider = None
    if openai_key:
        provider = "openai"
        print("[Embedding] Phát hiện API Key OpenAI. Sử dụng OpenAI API.")
    elif gemini_key:
        provider = "gemini"
        print("[Embedding] Phát hiện API Key Gemini. Sử dụng Gemini API.")
    else:
        provider = "mock"
        print("[Embedding] [Chế độ giả lập] Không tìm thấy API Key thực tế. Sử dụng Mock Local Embeddings.")

    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(docs_dir, "**/*.txt"), recursive=True) + glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

    if not pdf_files and not txt_files:
        print("[Thông báo] Thư mục docs/ trống rỗng. Tự động tạo tệp PDF mẫu.")
        sample_pdf_path = os.path.join(docs_dir, "smartchef_manual.pdf")
        generate_sample_pdf(sample_pdf_path)
        pdf_files = [sample_pdf_path]

    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    documents = []
    print("\n--- Đang tải các tài liệu từ thư mục docs/ ---")
    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path}")
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Lỗi khi đọc PDF {pdf_path}: {e}")

    for txt_path in txt_files:
        print(f"Loading Text: {txt_path}")
        try:
            txt_loader = TextLoader(txt_path, encoding="utf-8")
            documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Lỗi khi đọc Text/Markdown {txt_path}: {e}")

    print(f"Tổng số trang/tài liệu đã đọc: {len(documents)}")
    if not documents:
        print("Không có tài liệu nào được tải thành công!")
        sys.exit(1)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("\n--- Đang thực hiện tách nhỏ văn bản (Split) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Tổng số chunks sau khi tách: {len(chunks)}")

    if chunks:
        print(f"Chunk mẫu thứ nhất (độ dài {len(chunks[0].page_content)} ký tự):")
        print("-" * 50)
        print(chunks[0].page_content)
        print("-" * 50)
        print(f"Metadata: {chunks[0].metadata}")

    print("\n--- Đang khởi tạo mô hình Embedding ---")
    embeddings: Embeddings
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        model_name = "text-embedding-3-small"
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_key
        )
        model_name = "models/text-embedding-004"
    else:
        embeddings = MockEmbeddings()
        model_name = "mock-local-embeddings-1536"

    from langchain_chroma import Chroma

    persist_dir = "chroma_db"

    # Xoá db cũ để tránh trùng lặp hoặc sai lệch dữ liệu cũ phi-đơn-trị
    if os.path.exists(persist_dir):
        print(f"Phát hiện database cũ. Đang dọn dẹp thư mục: {persist_dir}")
        import shutil
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Không thể xoá thư mục {persist_dir}: {e}")

    print(f"\n--- Đang lưu trữ các chunks vào Chroma tại: {persist_dir} ---")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    meta_path = os.path.join(persist_dir, "embedding_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "provider": provider,
            "model_name": model_name
        }, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu trữ {len(chunks)} chunks thành công và tạo file meta tại {meta_path}.")
    print("Quy trình indexing hoàn tất thành công!")

if __name__ == "__main__":
    main()
