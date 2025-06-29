# /tools/search/hybrid_search.py
"""
Tool để thực hiện tìm kiếm lai (hybrid search), kết hợp nhiều chiến lược tìm kiếm,
tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Nhập các công cụ tìm kiếm khác để kết hợp
from tools.search.vector_search import VectorSearchTool
from tools.search.graph_search import GraphSearchTool

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchTool:
    """
    Một class chứa các công cụ để kết hợp các phương pháp tìm kiếm khác nhau
    nhằm mang lại kết quả tốt nhất.
    """
    def __init__(self, vector_tool: VectorSearchTool, graph_tool: GraphSearchTool):
        """
        Khởi tạo tool với các instance của các công cụ tìm kiếm phụ thuộc.
        Cách tiếp cận này được gọi là Dependency Injection, giúp cho việc kiểm thử dễ dàng hơn.

        Args:
            vector_tool (VectorSearchTool): Instance của công cụ tìm kiếm vector.
            graph_tool (GraphSearchTool): Instance của công cụ tìm kiếm đồ thị.
        """
        self.vector_search_tool = vector_tool
        self.graph_search_tool = graph_tool
        logger.info("HybridSearchTool đã được khởi tạo với các công cụ tìm kiếm phụ thuộc.")

    @tool(
        name="multi_modal_search",
        description="Thực hiện tìm kiếm đa phương thức bằng cách kết hợp tìm kiếm vector, đồ thị và từ khóa.",
        cache_results=False  # Không cache ở cấp này, để các tool con tự cache
    )
    def multi_modal_search(
        self,
        query: str,
        query_vector: List[float],
        entities: List[str],
        search_types: List[str] = ["vector", "graph"]
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm trên nhiều phương thức và hợp nhất kết quả.

        Args:
            query (str): Chuỗi truy vấn gốc của người dùng.
            query_vector (List[float]): Vector embedding của truy vấn.
            entities (List[str]): Danh sách các thực thể được trích xuất từ truy vấn.
            search_types (List[str]): Các loại tìm kiếm cần thực hiện.

        Returns:
            List[Dict[str, Any]]: Một danh sách kết quả đã được hợp nhất và xếp hạng.
        """
        logger.info(f"Bắt đầu multi-modal search cho query: '{query}'")
        all_results = {}

        # Thực hiện các loại tìm kiếm được yêu cầu
        if "vector" in search_types:
            try:
                # Sử dụng hybrid_search của VectorSearchTool để kết hợp vector và text
                vector_results = self.vector_search_tool.hybrid_search(
                    query_text=query,
                    query_vector=query_vector
                )
                all_results["vector"] = vector_results
            except Exception as e:
                logger.error(f"Lỗi trong quá trình vector search: {e}")
                all_results["vector"] = []

        if "graph" in search_types and entities:
            try:
                graph_results = self.graph_search_tool.search_graph(entities=entities)
                all_results["graph"] = graph_results
            except Exception as e:
                logger.error(f"Lỗi trong quá trình graph search: {e}")
                all_results["graph"] = []
        
        # Hợp nhất và xếp hạng kết quả
        merged_results = self.rank_and_merge_results(all_results)
        logger.info(f"Hoàn tất multi-modal search, trả về {len(merged_results)} kết quả.")
        
        return merged_results

    def rank_and_merge_results(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Hàm nội bộ để hợp nhất và xếp hạng kết quả từ các nguồn khác nhau.
        Đây không phải là một 'tool'.

        Args:
            results (Dict[str, List]): Một dict chứa kết quả từ mỗi loại tìm kiếm.

        Returns:
            List[Dict[str, Any]]: Danh sách kết quả cuối cùng.
        """
        final_results = {}
        
        # Logic xếp hạng và hợp nhất có thể rất phức tạp.
        # Ví dụ đơn giản: gán điểm và kết hợp.
        
        # Gán điểm cho kết quả vector search dựa trên _distance
        for res in results.get("vector", []):
            video_id = res.get("video_id", str(res.get("id"))) # Giả sử có video_id hoặc id
            if video_id not in final_results:
                final_results[video_id] = {"score": 0, "sources": [], "data": res}
            # Điểm càng cao nếu distance càng thấp
            final_results[video_id]["score"] += 1.0 / (res.get("_distance", 1.0) + 0.1)
            final_results[video_id]["sources"].append("vector")

        # Cộng điểm cho kết quả từ graph search
        for res in results.get("graph", []):
            video_id = res.get("video_id", str(res.get("id")))
            if video_id not in final_results:
                final_results[video_id] = {"score": 0, "sources": [], "data": res}
            final_results[video_id]["score"] += 0.5 # Cộng một điểm cố định
            final_results[video_id]["sources"].append("graph")
            
        # Sắp xếp kết quả dựa trên điểm số
        sorted_results = sorted(final_results.values(), key=lambda x: x["score"], reverse=True)
        
        # Trả về dữ liệu gốc của các kết quả đã sắp xếp
        return [item["data"] for item in sorted_results]
