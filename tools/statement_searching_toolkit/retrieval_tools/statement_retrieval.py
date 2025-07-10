# ============================
# 📦 IMPORTS - Giải thích mục đích từng gói
# ============================
from typing import List, Optional  # 👉 Kiểu dữ liệu cho type hinting
from agno.tools import tool        # 👉 Decorator để đăng ký tool cho agno framework
import logging                     # 👉 Ghi log cho quá trình xử lý, debug
import chromadb                    # 👉 Thư viện thao tác với vector database Chroma
from chromadb.config import Settings  # 👉 Cấu hình cho ChromaDB
from sentence_transformers import SentenceTransformer  # 👉 Model embedding câu
import numpy as np                 # 👉 Xử lý mảng số, chuyển đổi embedding
import re                          # 👉 Xử lý chuỗi, tách câu từ mô tả
from transformers import pipeline  # 👉 Sử dụng transformers miễn phí từ HuggingFace
# ============================
# ⚙️ CLASS DEFINITION
# ============================
class StatementRetrievalTools:
    """
    🔍 Bộ công cụ truy vấn statement cho agno: trích xuất statement từ mô tả, embedding, và truy vấn tương tự trong ChromaDB.
    """
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", summarizer_model_name: str = "facebook/bart-large-cnn"):
        """
        Khởi tạo tool, logger và sentence-transformers model.
        Args:
            embedding_model_name: Tên model embedding cho sentence-transformers
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = embedding_model_name

        self.embedding_model = SentenceTransformer(embedding_model_name)
        # 🧠 Khởi tạo pipeline summarizer một lần duy nhất cho toàn bộ class
        self.summarizer = pipeline("summarization", model=summarizer_model_name)

    @tool(
        name="extract_statements_from_description",
        description="Nhận vào một câu mô tả frame ảnh (natural language), trích xuất các statement đơn giản từ đó.",
        cache_results=False
    )
    def extract_statements_from_description(self, description: str) -> List[str]:
        """
        Trích xuất các statement đơn giản từ mô tả ngôn ngữ tự nhiên về frame ảnh bằng mô hình ngôn ngữ (LLM).
        Args:
            description: Chuỗi mô tả frame ảnh
        Returns:
            List[str]: Danh sách statement đơn giản
        """
        try:
            # 🧠 Sử dụng mô hình ngôn ngữ miễn phí (ví dụ: HuggingFace Transformers - model nhỏ như distilbert, bart, hoặc local LLM)
            # 📝 Prompt hướng dẫn mô hình tóm tắt thành các statement ngắn gọn, rõ nghĩa
            prompt = (
                "Hãy phân tích đoạn mô tả sau và sinh ra các statement ngắn gọn, mỗi statement mô tả về một đối tượng đơn lẻ hoặc về mối quan hệ giữa hai đối tượng, chỉ dựa trên nội dung mô tả. "
                "Chỉ trả về danh sách các statement, không giải thích thêm.\n\n"
                f"Mô tả: {description}"
            )
            # Tóm tắt prompt (có thể dùng prompt + description để tăng tính định hướng)
            summary = self.summarizer(prompt, max_length=130, min_length=20, do_sample=False)[0]["summary_text"]
            # 📝 Tách các statement dựa trên dấu chấm, xuống dòng
            statements = [s.strip('-•* ') for s in re.split(r'[.\n]+', summary) if len(s.strip()) > 5]
            self.logger.info(f"Extracted {len(statements)} statements from description (transformers).")
            return statements
        except Exception as e:
            self.logger.error(f"Lỗi khi tách statement bằng transformers: {e}")
            # Fallback: tách câu đơn giản nếu LLM lỗi
            statements = re.split(r'[.;\n]+', description)
            statements = [s.strip() for s in statements if len(s.strip()) > 5]
            return statements

    @tool(
        name="embed_statements",
        description="Nhận vào danh sách statement (list[str]), trả về list embedding vector tương ứng.",
        cache_results=False
    )
    def embed_statements(self, statements: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Nhúng các statement thành vector embedding bằng sentence-transformers.
        Args:
            statements: List[str] - Danh sách statement cần embedding
            model_name: Tên model embedding (nếu None sẽ dùng model đã khởi tạo)
        Returns:
            List[List[float]] - Danh sách embedding vector
        """
        model = self.embedding_model if model_name is None or model_name == self.embedding_model_name else SentenceTransformer(model_name)
        self.logger.info(f"Embedding {len(statements)} statements bằng model: {model_name or self.embedding_model_name}")
        try:
            embeddings = model.encode(statements, show_progress_bar=False, convert_to_numpy=True)
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        except Exception as e:
            self.logger.error(f"Lỗi khi embedding statements: {e}")
            return []

    @tool(
        name="search_similar_frames",
        description="Tìm top-k frame có embedding tương tự nhất trong ChromaDB dựa trên embedding đầu vào.",
        cache_results=False
    )
    def search_similar_frames(self, query_embeddings: List[List[float]], k: int = 5, collection_name: str = "statements") -> List[dict]:
        """
        Tìm top-k frame có embedding tương tự nhất trong ChromaDB dựa trên embedding đầu vào.
        Args:
            query_embeddings: List[List[float]] - Danh sách embedding truy vấn
            k: Số lượng kết quả top-k
            collection_name: Tên collection trong Chroma
        Returns:
            List[dict]: Danh sách kết quả (metadata + score)
        """
        try:
            client = chromadb.Client(Settings(persist_directory="./chroma_data"))
            collection = client.get_collection(collection_name)
            np_queries = np.array(query_embeddings, dtype=np.float32)
            # 📝 Truy vấn từng embedding, lấy top-k cho mỗi embedding
            results = []
            for q_emb in np_queries:
                res = collection.query(query_embeddings=[q_emb], n_results=k, include=["metadatas", "distances", "documents"])
                ids = res.get("ids")
                ids = ids[0] if ids and isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list) else (ids or [])
                distances = res.get("distances")
                distances = distances[0] if distances and isinstance(distances, list) and len(distances) > 0 and isinstance(distances[0], list) else (distances or [])
                metadatas = res.get("metadatas")
                metadatas = metadatas[0] if metadatas and isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list) else (metadatas or [])
                documents = res.get("documents")
                documents = documents[0] if documents and isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list) else (documents or [])
                for i in range(len(ids)):
                    results.append({
                        "id": ids[i],
                        "score": distances[i] if i < len(distances) else None,
                        "metadata": metadatas[i] if i < len(metadatas) else None,
                        "document": documents[i] if i < len(documents) else None
                    })
            # Sắp xếp theo score tăng dần (gần nhất)
            results = sorted(results, key=lambda x: x["score"] if x["score"] is not None else float('inf'))
            self.logger.info(f"Found {len(results)} similar frames.")
            return results[:k]
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn ChromaDB: {e}")
            return []
