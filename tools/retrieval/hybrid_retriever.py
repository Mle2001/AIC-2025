# /tools/retrieval/hybrid_retriever.py
"""
Tool để thực hiện truy xuất lai (hybrid retrieval), kết hợp nhiều chiến lược,
tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Nhập các công cụ truy xuất khác để kết hợp
from tools.retrieval.vector_retriever import VectorRetrieverTool
from tools.retrieval.graph_retriever import GraphRetrieverTool

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRetrieverTool:
    """
    Một class chứa các công cụ để kết hợp các phương pháp truy xuất khác nhau
    nhằm mang lại kết quả tốt nhất.
    """
    def __init__(self, vector_tool: VectorRetrieverTool, graph_tool: GraphRetrieverTool):
        """
        Khởi tạo tool với các instance của các công cụ truy xuất phụ thuộc.
        Sử dụng Dependency Injection để dễ dàng kiểm thử.

        Args:
            vector_tool (VectorRetrieverTool): Instance của công cụ truy xuất vector.
            graph_tool (GraphRetrieverTool): Instance của công cụ truy xuất đồ thị.
        """
        self.vector_retriever_tool = vector_tool
        self.graph_retriever_tool = graph_tool
        logger.info("HybridRetrieverTool đã được khởi tạo với các công cụ con.")

    @tool(
        name="multi_modal_retrieval",
        description="Thực hiện truy xuất đa phương thức bằng cách kết hợp truy xuất vector, đồ thị và từ khóa.",
        cache_results=False # Không cache ở cấp này, để các tool con tự cache
    )
    async def multi_modal_retrieval(
        self,
        query_text: str,
        query_vector: List[float],
        entities: List[Dict[str, Any]],
        search_types: List[str] = ["vector", "graph"]
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện truy xuất trên nhiều phương thức và hợp nhất kết quả.

        Args:
            query_text (str): Chuỗi truy vấn gốc của người dùng.
            query_vector (List[float]): Vector embedding của truy vấn.
            entities (List[Dict[str, Any]]): Danh sách các thực thể được trích xuất từ truy vấn.
                                             Mỗi thực thể là một dict, ví dụ: {'label': 'Person', 'name': 'John'}
            search_types (List[str]): Các loại tìm kiếm cần thực hiện.

        Returns:
            List[Dict[str, Any]]: Một danh sách kết quả đã được hợp nhất và xếp hạng.
        """
        logger.info(f"Bắt đầu multi-modal retrieval cho query: '{query_text}'")
        
        tasks = []
        # Chuẩn bị các tác vụ truy xuất bất đồng bộ
        if "vector" in search_types:
            tasks.append(self.vector_retriever_tool.hybrid_retrieval(
                query_text=query_text,
                query_vector=query_vector
            ))
        if "graph" in search_types and entities:
            # Có thể có nhiều truy vấn đồ thị phức tạp ở đây
            for entity in entities:
                tasks.append(self.graph_retriever_tool.retrieve_entities_by_property(
                    entity_label=entity.get("label", "Unknown"),
                    properties={"name": entity.get("name")}
                ))

        # Chạy các tác vụ song song và thu thập kết quả
        all_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả và lỗi
        vector_results = []
        graph_results = []
        task_index = 0
        if "vector" in search_types:
            if not isinstance(all_results_list[task_index], Exception):
                vector_results = all_results_list[task_index]
            else:
                logger.error(f"Lỗi trong vector retrieval: {all_results_list[task_index]}")
            task_index += 1
        
        if "graph" in search_types and entities:
            for _ in entities:
                if not isinstance(all_results_list[task_index], Exception):
                    graph_results.extend(all_results_list[task_index])
                else:
                     logger.error(f"Lỗi trong graph retrieval: {all_results_list[task_index]}")
                task_index += 1

        # Hợp nhất và xếp hạng kết quả
        merged_results = self._rank_and_merge_results({
            "vector": vector_results,
            "graph": graph_results
        })
        logger.info(f"Hoàn tất multi-modal retrieval, trả về {len(merged_results)} kết quả.")
        
        return merged_results

    def _rank_and_merge_results(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Hàm nội bộ để hợp nhất và xếp hạng kết quả từ các nguồn khác nhau.
        """
        final_results = {}
        
        # Gán điểm cho kết quả vector search dựa trên _distance
        for res in results.get("vector", []):
            # Giả sử mỗi kết quả có một ID duy nhất, ví dụ 'video_id' hoặc 'doc_id'
            item_id = res.get("video_id") or res.get("id")
            if not item_id: continue

            if item_id not in final_results:
                final_results[item_id] = {"score": 0, "sources": [], "data": res}
            # Điểm càng cao nếu distance càng thấp. Thêm 1.0 để nhấn mạnh tầm quan trọng.
            final_results[item_id]["score"] += 1.0 + (1.0 / (res.get("_distance", 1.0) + 0.1))
            final_results[item_id]["sources"].append("vector")

        # Cộng điểm cho kết quả từ graph search
        for res in results.get("graph", []):
            item_id = res.get("video_id") or res.get("id")
            if not item_id: continue
            
            if item_id not in final_results:
                final_results[item_id] = {"score": 0, "sources": [], "data": res}
            # Cộng một điểm cố định cho mỗi lần xuất hiện trong kết quả đồ thị
            final_results[item_id]["score"] += 0.8 
            final_results[item_id]["sources"].append("graph")
            
        # Sắp xếp kết quả dựa trên điểm số từ cao đến thấp
        sorted_results = sorted(final_results.values(), key=lambda x: x["score"], reverse=True)
        
        # Trả về dữ liệu gốc của các kết quả đã sắp xếp
        return [item["data"] for item in sorted_results]
