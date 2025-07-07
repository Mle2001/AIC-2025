# /tools/retrieval/vector_retriever.py
"""
Công cụ truy xuất vector chuyên dụng.
"""
import logging
from typing import List, Dict, Any, Optional

import lancedb
import numpy as np
from agno.tools import tool

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Tập hợp các công cụ để truy xuất dữ liệu từ LanceDB, kế thừa từ BaseRetriever.
    """

    def __init__(self, db_path: str = "./data/lancedb"):
        """
        Khởi tạo công cụ và đường dẫn DB. Việc kết nối sẽ được gọi bởi super().__init__().
        """
        self.db_path = db_path
        self._db_conn = None
        super().__init__()

    def _connect(self):
        """Thiết lập kết nối tới cơ sở dữ liệu LanceDB."""
        try:
            self._db_conn = lancedb.connect(self.db_path)
            logger.info(f"Đã kết nối thành công tới LanceDB tại: {self.db_path}")
        except Exception as e:
            logger.error(f"Không thể kết nối tới LanceDB tại '{self.db_path}': {e}", exc_info=True)
            raise ConnectionError(f"Lỗi kết nối LanceDB: {e}") from e

    @tool(
        name="retrieve_scenes_by_vector",
        description="Truy xuất các cảnh (scenes) tương đồng về mặt ngữ nghĩa bằng một vector truy vấn trên một cột vector cụ thể.",
        cache_results=True,
        cache_ttl=300
    )
    def retrieve_by_vector(
        self,
        query_vector: List[float],
        target_vector_column: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện truy xuất tương đồng trên một cột vector cụ thể.
        """
        if not self._db_conn:
            logger.error("Kết nối LanceDB chưa được thiết lập. Không thể truy xuất.")
            return []
            
        try:
            scenes_table = self._db_conn.open_table("scenes")
            search_query = scenes_table.search(
                np.array(query_vector, dtype=np.float32),
                vector_column_name=target_vector_column
            ).limit(top_k)

            if filters:
                for key, value in filters.items():
                    search_query = search_query.where(f"{key} = '{value}'")
            
            results = search_query.select([
                "scene_id", "video_id", "start_seconds", "end_seconds", "raw_ocr_text"
            ]).to_list()
            
            logger.info(f"Thực hiện truy xuất trên cột '{target_vector_column}' và tìm thấy {len(results)} kết quả.")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong quá trình truy xuất vector: {e}", exc_info=True)
            return []

    @tool(
        name="retrieve_scenes_hybrid",
        description="Kết hợp truy xuất ngữ nghĩa (vector) và truy xuất từ khóa (full-text) để tăng độ chính xác.",
        cache_results=True,
        cache_ttl=300
    )
    def retrieve_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện truy xuất lai, kết hợp vector và từ khóa.
        """
        if not self._db_conn:
            logger.error("Kết nối LanceDB chưa được thiết lập. Không thể truy xuất.")
            return []
            
        try:
            scenes_table = self._db_conn.open_table("scenes")
            results = scenes_table.search(
                np.array(query_vector, dtype=np.float32),
                vector_column_name="visual_embedding"
            ).where(f"raw_ocr_text LIKE '%{query_text}%'").limit(top_k).to_list()

            logger.info(f"Thực hiện truy xuất lai cho '{query_text}' và tìm thấy {len(results)} kết quả.")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong quá trình truy xuất lai: {e}", exc_info=True)
            return []
