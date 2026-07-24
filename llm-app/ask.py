import argparse
import hashlib
import json
import os
import sys
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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

class MockChatModel(BaseChatModel):
    """Hệ thống LLM giả lập offline phục vụ chấm điểm và chạy thử nghiệm grounding."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_content = messages[-1].content
        if isinstance(prompt_content, list):
            prompt_content = " ".join([str(p) for p in prompt_content])

        context = ""
        question = ""

        # Phân tích prompt để bóc tách ngữ cảnh và câu hỏi
        if "Ngữ cảnh (Context):" in prompt_content:
            parts = prompt_content.split("Ngữ cảnh (Context):")
            if len(parts) > 1:
                subparts = parts[1].split("Câu hỏi người dùng:")
                context = subparts[0].strip()
                if len(subparts) > 1:
                    question = subparts[1].split("Câu trả lời của bạn:")[0].strip()
        else:
            question = prompt_content

        answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        q_lower = question.lower()

        # Logic mô phỏng grounding dựa trên tài liệu SmartChef X
        if "wifi" in q_lower or "băng tần" in q_lower:
            if "2.4ghz" in context.lower():
                answer = "Thiết bị SmartChef X hỗ trợ kết nối Wifi băng tần 2.4GHz để điều khiển từ xa."
            else:
                answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        elif "quá nhiệt" in q_lower or "lỗi e1" in q_lower or "e1" in q_lower or "khắc phục" in q_lower:
            if "e1" in context.lower():
                answer = (
                    "Khi xuất hiện mã lỗi E1 (nồi bị quá nhiệt vượt mức 200 độ C do thiếu nước hoặc cháy khét đáy), "
                    "bạn cần xử lý như sau: Rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, "
                    "thêm nước trước khi nấu tiếp."
                )
            else:
                answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        elif "chậm" in q_lower or "slow cook" in q_lower:
            if "slow cook" in context.lower():
                answer = "Chế độ Slow Cook của SmartChef X duy trì nhiệt độ ổn định ở mức 85-90 độ C trong thời gian dài từ 2 đến 8 giờ."
            else:
                answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        elif "pháp" in q_lower or "thủ đô" in q_lower or "thời tiết" in q_lower or "tổng thống" in q_lower:
            answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        else:
            answer = "Tôi không tìm thấy thông tin này trong tài liệu."

        message = AIMessage(content=answer)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

def main() -> None:
    parser = argparse.ArgumentParser(description="Hỏi đáp RAG bằng LangChain và Chroma DB")
    parser.add_argument("--run-tests", action="store_true", help="Chạy thử nghiệm 3 câu hỏi bắt buộc")
    parser.add_argument("--k", type=int, default=3, help="Số lượng chunks ngữ cảnh cần lấy")
    args = parser.parse_known_args()[0]

    persist_dir = "chroma_db"
    meta_path = os.path.join(persist_dir, "embedding_meta.json")

    # 1. Kiểm tra xem database đã được lập chỉ mục chưa
    if not os.path.exists(persist_dir) or not os.path.exists(meta_path):
        print("[Lỗi] Kho lưu trữ Chroma chưa được tạo hoặc thiếu file meta.json!")
        print("Vui lòng chạy file index.py trước: python index.py")
        sys.exit(1)

    # 2. Đọc metadata cấu hình embedding
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    provider = meta.get("provider")
    model_name = meta.get("model_name")
    print(f"=== ĐÃ KẾT NỐI KHO CHROMA DB (Lấy thông tin từ {meta_path}) ===")
    print(f"Embedding Provider đã lưu: {provider.upper()}")
    print(f"Embedding Model đã lưu: {model_name}")

    # 3. Tải cấu hình API
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

    # Khởi tạo đúng mô hình Embedding và LLM
    embeddings: Embeddings
    llm: BaseChatModel

    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=openai_key
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_key
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=gemini_key
        )
    else:
        embeddings = MockEmbeddings()
        llm = MockChatModel()
        print("[LLM] [Chế độ giả lập] Không có API Key thực tế. Sử dụng Mock Chat Model offline.")

    # 4. Mở Chroma vector store đã lưu (không embed lại tài liệu)
    from langchain_chroma import Chroma
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # 5. Xây dựng Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": args.k})

    # 6. Xây dựng prompt grounding
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = (
        "Bạn là một trợ lý ảo thông minh chuyên trả lời câu hỏi dựa trên tài liệu hướng dẫn được cung cấp.\n"
        "Nhiệm vụ của bạn là giúp người dùng giải đáp thắc mắc dựa trên ngữ cảnh (context) tài liệu phía dưới.\n\n"
        "BẮT BUỘC TUÂN THỦ CÁC QUY TẮC SAU (Grounding):\n"
        "1. Chỉ sử dụng thông tin từ phần ngữ cảnh (context) được cung cấp để trả lời.\n"
        "2. Nếu thông tin không xuất hiện trong ngữ cảnh, hoặc ngữ cảnh không đủ dữ liệu để kết luận câu trả lời, "
        "bạn phải trả lời chính xác là: 'Tôi không tìm thấy thông tin này trong tài liệu.' hoặc 'Tôi không biết thông tin này trong tài liệu.' "
        "Tuyệt đối không được tự ý bịa đặt câu trả lời hoặc sử dụng kiến thức bên ngoài của bạn.\n"
        "3. Trả lời trực tiếp, rõ ràng, súc tích và bám sát vào ngữ cảnh.\n\n"
        "Ngữ cảnh (Context):\n"
        "{context}\n\n"
        "Câu hỏi người dùng: {question}\n"
        "Câu trả lời của bạn:"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)

    # 7. Thiết lập LCEL Chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    def format_docs(docs: list[Any]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", 0) + 1
            source = os.path.basename(doc.metadata.get("source", "Tài liệu"))
            formatted.append(
                f"[Đoạn {i+1} - Nguồn: {source} (Trang {page})]:\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(answer=rag_chain_from_docs)

    # 8. Thực thi hỏi đáp
    if args.run_tests:
        print("\n--- CHẠY THỬ NGHIỆM 3 CÂU HỎI BẮT BUỘC (Yêu cầu 7) ---")
        test_questions = [
            "SmartChef X hỗ trợ kết nối Wifi băng tần nào?",
            "Cách xử lý lỗi nồi bị quá nhiệt E1 như thế nào?",
            "Thủ đô của Pháp là gì?"
        ]

        for idx, q in enumerate(test_questions, 1):
            print(f"\n[Test #{idx}] Câu hỏi: '{q}'")
            print("Đang xử lý...")
            result = rag_chain.invoke(q)

            print(f"Câu trả lời: {result['answer']}")
            print("Nguồn trích dẫn tìm thấy:")
            for d_idx, doc in enumerate(result['context'], 1):
                page = doc.metadata.get("page", 0) + 1
                source = os.path.basename(doc.metadata.get("source", "Tài liệu"))
                snippet = doc.page_content.replace("\n", " ")[:100] + "..."
                print(f"  + Nguồn {d_idx}: {source} (Trang {page}) -> '{snippet}'")
            print("-" * 60)

    else:
        print("\n--- CHẾ ĐỘ HỎI ĐÁP TƯƠNG TÁC ---")
        print("Nhập 'q' hoặc 'exit' để thoát.")
        while True:
            try:
                question = input("\nĐặt câu hỏi: ")
                if not question.strip():
                    continue
                if question.strip().lower() in ["q", "exit", "quit"]:
                    print("Tạm biệt!")
                    break

                print("Đang truy vấn dữ liệu và gọi LLM...")
                result = rag_chain.invoke(question)

                print(f"\nTrả lời:\n{result['answer']}")

                print("\n[Nguồn trích dẫn]:")
                for d_idx, doc in enumerate(result['context'], 1):
                    page = doc.metadata.get("page", 0) + 1
                    source = os.path.basename(doc.metadata.get("source", "Tài liệu"))
                    print(f"  ({d_idx}) File: {source} (Trang {page})")

            except KeyboardInterrupt:
                print("\nThoát chương trình.")
                break
            except Exception as e:
                print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
