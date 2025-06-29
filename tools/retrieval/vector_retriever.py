# /tools/retrieval/vector_retriever.py
"""
Tool cho vector retrieval sử dụng LanceDB, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any, Optional

from agno.tools import tool
import lancedb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorRetrieverTool:
    """
    Class chứa các công cụ để truy xuất thông tin từ LanceDB.
    """
    def __init__(self, db_path: str = "./data/lancedb"):
        try:
            logger.info(f"Đang kết nối tới LanceDB tại: {db_path}")
            self.db = lancedb.connect(db_path)
        except Exception as e:
            logger.error(f"Lỗi khi kết nối tới LanceDB: {e}", exc_info=True)
            raise ConnectionError(f"Không thể kết nối tới LanceDB: {e}")

    @tool(
        name="retrieve_by_vector",
        description="Truy xuất các vector tương tự trong một bảng của LanceDB.",
        cache_results=True,
        cache_ttl=300
    )
    def retrieve_by_vector(
        self,
        query_vector: List[float],
        table_name: str = "video_embeddings",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        logger.info(f"Thực hiện vector retrieval trên bảng '{table_name}' với limit={limit}.")
        try:
            table = self.db.open_table(table_name)
            results = table.search(query_vector).limit(limit).to_list()
            logger.info(f"Truy xuất được {len(results)} kết quả.")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện vector retrieval: {e}", exc_info=True)
            raise e

    @tool(
        name="hybrid_retrieval",
        description="Thực hiện truy xuất lai, kết hợp vector và từ khóa (full-text search).",
        cache_results=True,
        cache_ttl=300
    )
    def hybrid_retrieval(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        table_name: str = "video_embeddings",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        logger.info(f"Thực hiện hybrid retrieval trên bảng '{table_name}' với query='{query_text}'.")
        try:
            table = self.db.open_table(table_name)
            search_query = table.search(query_vector, query_type="hybrid").query(query_text)
            results = search_query.limit(limit).to_list()
            logger.info(f"Truy xuất được {len(results)} kết quả từ hybrid retrieval.")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện hybrid retrieval: {e}", exc_info=True)
            raise e
