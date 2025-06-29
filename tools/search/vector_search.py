# /tools/search/vector_search.py
"""
Tool cho vector search trong LanceDB, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any, Optional

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện LanceDB
import lancedb

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorSearchTool:
    """
    Một class chứa các công cụ để thực hiện tìm kiếm vector và tìm kiếm lai
    trên cơ sở dữ liệu LanceDB.
    """
    def __init__(self, db_path: str = "./data/lancedb"):
        """
        Khởi tạo tool và kết nối tới cơ sở dữ liệu LanceDB.

        Args:
            db_path (str): Đường dẫn đến thư mục chứa dữ liệu của LanceDB.
        """
        try:
            logger.info(f"Đang kết nối tới LanceDB tại: {db_path}")
            self.db = lancedb.connect(db_path)
            logger.info("Kết nối LanceDB thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi kết nối tới LanceDB tại '{db_path}': {e}", exc_info=True)
            raise ConnectionError(f"Không thể kết nối tới LanceDB: {e}")

    @tool(
        name="vector_search",
        description="Tìm kiếm các vector tương tự trong một bảng của LanceDB.",
        cache_results=True,
        cache_ttl=300 # Cache kết quả tìm kiếm trong 5 phút
    )
    def search_vectors(
        self,
        query_vector: List[float],
        table_name: str = "video_embeddings",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm dựa trên sự tương đồng của vector.

        Args:
            query_vector (List[float]): Vector truy vấn để tìm kiếm.
            table_name (str): Tên của bảng trong LanceDB cần tìm kiếm.
            limit (int): Số lượng kết quả tối đa cần trả về.

        Returns:
            List[Dict[str, Any]]: Danh sách các kết quả phù hợp nhất, bao gồm cả điểm tương đồng.
        """
        logger.info(f"Thực hiện vector search trên bảng '{table_name}' với limit={limit}.")
        try:
            table = self.db.open_table(table_name)
            results = table.search(query_vector).limit(limit).to_list()
            logger.info(f"Tìm thấy {len(results)} kết quả.")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện vector search trên bảng '{table_name}': {e}", exc_info=True)
            raise e

    @tool(
        name="hybrid_vector_search",
        description="Thực hiện tìm kiếm lai, kết hợp tìm kiếm vector và tìm kiếm từ khóa (full-text search).",
        cache_results=True,
        cache_ttl=300
    )
    def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        table_name: str = "video_embeddings",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Kết hợp tìm kiếm vector (nếu có) và tìm kiếm từ khóa để có kết quả chính xác hơn.
        Yêu cầu bảng đã được tạo chỉ mục FTS (Full-Text Search).

        Args:
            query_text (str): Chuỗi văn bản truy vấn cho phần full-text search.
            query_vector (Optional[List[float]]): Vector truy vấn (tùy chọn).
            table_name (str): Tên của bảng cần tìm kiếm.
            limit (int): Số lượng kết quả tối đa.

        Returns:
            List[Dict[str, Any]]: Danh sách các kết quả phù hợp.
        """
        logger.info(f"Thực hiện hybrid search trên bảng '{table_name}' với query='{query_text}'.")
        try:
            table = self.db.open_table(table_name)
            
            # LanceDB xử lý việc có vector hay không một cách tự động
            search_query = table.search(query_vector, query_type="hybrid").query(query_text)
            results = search_query.limit(limit).to_list()

            logger.info(f"Tìm thấy {len(results)} kết quả từ hybrid search.")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện hybrid search trên bảng '{table_name}': {e}", exc_info=True)
            raise e
