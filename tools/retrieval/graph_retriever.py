# /tools/retrieval/graph_retriever.py
import logging
from typing import List, Dict, Any, Optional
from agno.tools import tool
from neo4j import AsyncGraphDatabase, Driver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRetrieverTool:
    def __init__(self, uri: str, username: str, password: str):
        self._driver: Optional[Driver] = None
        try:
            logger.info(f"Đang tạo driver kết nối tới Neo4j tại: {uri}")
            self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        except Exception as e:
            logger.error(f"Lỗi khi tạo driver Neo4j: {e}", exc_info=True)
            raise ConnectionError(f"Không thể tạo driver Neo4j: {e}")

    async def close(self):
        if self._driver:
            await self._driver.close()
            logger.info("Đã đóng kết nối Neo4j.")
            
    async def _verify_connectivity(self):
        if not self._driver:
             raise ConnectionError("Driver Neo4j chưa được khởi tạo.")
        try:
            await self._driver.verify_connectivity()
            logger.info("Kết nối Neo4j thành công.")
        except Exception as e:
            logger.error(f"Xác minh kết nối Neo4j thất bại: {e}", exc_info=True)
            raise ConnectionError(f"Không thể xác minh kết nối Neo4j: {e}")

    @tool(name="retrieve_entities_by_property", description="Truy xuất các nút (entities) trong đồ thị dựa trên nhãn và thuộc tính.", cache_results=True, cache_ttl=600)
    async def retrieve_entities_by_property(self, entity_label: str, properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._driver:
            raise ConnectionError("Driver Neo4j chưa được khởi tạo.")
        
        prop_str = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MATCH (n:{entity_label}) WHERE {prop_str} RETURN n LIMIT 25"
        
        async with self._driver.session() as session:
            result = await session.run(query, **properties)
            records = await result.data()
            nodes = [record["n"] for record in records]
            return nodes

    @tool(name="find_related_entities", description="Tìm các thực thể có liên quan đến một thực thể cho trước.", cache_results=True)
    async def find_related_entities(self, entity_name: str, entity_label: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        if not self._driver:
            raise ConnectionError("Driver Neo4j chưa được khởi tạo.")

        query = f"MATCH (startNode:{entity_label} {{name: $entity_name}})-[*1..{max_depth}]-(relatedNode) RETURN DISTINCT relatedNode LIMIT 50"
        
        async with self._driver.session() as session:
            result = await session.run(query, entity_name=entity_name)
            records = await result.data()
            related_nodes = [record["relatedNode"] for record in records]
            return related_nodes
