# /tools/retrieval/graph_retriever.py
"""
Công cụ truy xuất dữ liệu từ Knowledge Graph (Neo4j).
"""
import logging
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase, Driver
from agno.tools import tool

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class GraphRetriever(BaseRetriever):
    """
    Tập hợp các công cụ để truy xuất dữ liệu từ Neo4j, kế thừa từ BaseRetriever.
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """
        Khởi tạo công cụ và thông tin kết nối.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[Driver] = None
        super().__init__()

    def _connect(self):
        """Thiết lập kết nối tới cơ sở dữ liệu Neo4j."""
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            logger.info(f"Đã kết nối thành công tới Neo4j tại: {self.uri}")
        except Exception as e:
            logger.error(f"Không thể kết nối tới Neo4j tại '{self.uri}': {e}", exc_info=True)
            raise ConnectionError(f"Lỗi kết nối Neo4j: {e}") from e

    def close(self):
        """Đóng kết nối Neo4j khi không cần dùng nữa."""
        if self._driver:
            self._driver.close()
            logger.info("Đã đóng kết nối Neo4j.")

    @tool(
        name="find_scenes_by_entity",
        description="Tìm kiếm các cảnh (scenes) có chứa một thực thể (entity) cụ thể.",
        cache_results=True, cache_ttl=600
    )
    def find_scenes_by_entity(self, entity_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm các cảnh liên quan đến một thực thể cụ thể.
        """
        if not self._driver:
            logger.error("Kết nối Neo4j chưa được thiết lập.")
            return []

        query = """
        MATCH (s:Scene)-[:CONTAINS_ENTITY]->(e:Entity)
        WHERE toLower(e.name) = toLower($entity_name)
        RETURN s.scene_id AS scene_id, s.video_id AS video_id, 
               s.start_seconds AS start_seconds, s.end_seconds AS end_seconds
        LIMIT $top_k
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, entity_name=entity_name, top_k=top_k)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn graph cho thực thể '{entity_name}': {e}", exc_info=True)
            return []

    @tool(
        name="find_related_entities",
        description="Từ một thực thể đã biết, tìm các thực thể khác có liên quan trong cùng các cảnh.",
        cache_results=True, cache_ttl=600
    )
    def find_related_entities(self, entity_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Tìm các thực thể khác cùng xuất hiện trong các cảnh có chứa thực thể đầu vào.
        """
        if not self._driver:
            logger.error("Kết nối Neo4j chưa được thiết lập.")
            return []

        query = """
        MATCH (s:Scene)-[:CONTAINS_ENTITY]->(e1:Entity)
        WHERE toLower(e1.name) = toLower($entity_name)
        MATCH (s)-[:CONTAINS_ENTITY]->(e2:Entity)
        WHERE e1 <> e2
        RETURN e2.name AS related_entity, count(s) AS co_occurrence
        ORDER BY co_occurrence DESC
        LIMIT $top_k
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, entity_name=entity_name, top_k=top_k)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Lỗi khi tìm các thực thể liên quan cho '{entity_name}': {e}", exc_info=True)
            return []
