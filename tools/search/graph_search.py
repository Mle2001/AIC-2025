# /tools/search/graph_search.py
"""
Tool cho graph search trong Neo4j, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện Neo4j driver
from neo4j import GraphDatabase, Driver

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphSearchTool:
    """
    Một class chứa các công cụ để thực hiện tìm kiếm trên đồ thị tri thức Neo4j.
    """
    def __init__(self, uri: str, username: str, password: str):
        """
        Khởi tạo tool và driver kết nối tới Neo4j.

        Args:
            uri (str): URI của Neo4j instance (ví dụ: "bolt://localhost:7687").
            username (str): Tên đăng nhập.
            password (str): Mật khẩu.
        """
        self._driver: Optional[Driver] = None
        try:
            logger.info(f"Đang tạo driver kết nối tới Neo4j tại: {uri}")
            self._driver = GraphDatabase.driver(uri, auth=(username, password))
            self._driver.verify_connectivity()
            logger.info("Kết nối Neo4j thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi kết nối tới Neo4j: {e}", exc_info=True)
            raise ConnectionError(f"Không thể kết nối tới Neo4j: {e}")

    def close(self):
        """Đóng kết nối driver khi không cần dùng nữa."""
        if self._driver:
            self._driver.close()
            logger.info("Đã đóng kết nối Neo4j.")

    @tool(
        name="graph_search",
        description="Tìm kiếm các nút (entities) trong đồ thị tri thức dựa trên nhãn và thuộc tính.",
        cache_results=True,
        cache_ttl=600 # Cache kết quả trong 10 phút
    )
    def search_graph(
        self,
        entities: List[str],
        relationships: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện một truy vấn Cypher đơn giản để tìm các nút và mối quan hệ.
        Ví dụ: Tìm tất cả các nút 'Person' có tên là 'John Doe'.

        Args:
            entities (List[str]): Danh sách các thực thể cần tìm (ví dụ: ["Person", "Car"]).
            relationships (List[str]): Các mối quan hệ cần tìm (hiện tại chưa dùng, để mở rộng).

        Returns:
            List[Dict[str, Any]]: Danh sách các nút tìm được, mỗi nút là một dict chứa thuộc tính.
        """
        if not self._driver:
            raise ConnectionError("Driver Neo4j chưa được khởi tạo.")
            
        logger.info(f"Thực hiện graph search cho các thực thể: {entities}")
        
        # Ví dụ này sẽ tìm các nút có nhãn nằm trong danh sách entities
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN $entity_labels)
        RETURN n
        LIMIT 25
        """
        
        with self._driver.session() as session:
            results = session.run(query, entity_labels=entities)
            nodes = [record["n"]._properties for record in results]
            logger.info(f"Tìm thấy {len(nodes)} nút từ graph search.")
            return nodes

    @tool(
        name="find_related_entities",
        description="Tìm các thực thể có liên quan đến một thực thể cho trước trong một khoảng cách nhất định.",
        cache_results=True
    )
    def find_related_entities(self, entity: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Tìm các nút hàng xóm của một nút đã biết.

        Args:
            entity (str): Tên (hoặc ID) của thực thể gốc.
            max_depth (int): Độ sâu tối đa để tìm kiếm mối quan hệ.

        Returns:
            List[Dict[str, Any]]: Danh sách các thực thể liên quan.
        """
        if not self._driver:
            raise ConnectionError("Driver Neo4j chưa được khởi tạo.")

        logger.info(f"Tìm các thực thể liên quan đến '{entity}' với độ sâu={max_depth}")
        
        # Truy vấn tìm các nút liên quan đến nút có thuộc tính name là `entity`
        query = f"""
        MATCH (startNode {{name: $entity_name}})-[*1..{max_depth}]-(relatedNode)
        RETURN DISTINCT relatedNode
        LIMIT 50
        """
        
        with self._driver.session() as session:
            results = session.run(query, entity_name=entity)
            related_nodes = [record["relatedNode"]._properties for record in results]
            logger.info(f"Tìm thấy {len(related_nodes)} thực thể liên quan.")
            return related_nodes
