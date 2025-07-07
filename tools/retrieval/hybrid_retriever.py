# /tools/retrieval/hybrid_retriever.py
"""
Công cụ truy xuất lai (Hybrid Retriever).

Kết hợp kết quả từ nhiều nguồn truy xuất khác nhau (Vector, Graph)
để cung cấp một danh sách kết quả cuối cùng, đã được xếp hạng và tổng hợp.
"""
import logging
import asyncio
from typing import List, Dict, Any

from agno.tools import tool

# Import các công cụ truy xuất và xử lý khác
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from ..processors.text import TextFeatureExtractor

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Công cụ điều phối các phương thức truy xuất khác nhau và tổng hợp kết quả.
    Đây là công cụ chính mà RetrievalAgent sẽ sử dụng cho các truy vấn phức tạp.
    """

    def __init__(self):
        """Khởi tạo các retriever con và các công cụ cần thiết."""
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self.text_embedder = TextFeatureExtractor()
        logger.info("HybridRetriever đã được khởi tạo với các retriever con.")

    @tool(
        name="retrieve_comprehensive_scenes",
        description="Thực hiện một truy vấn phức hợp, kết hợp tìm kiếm ngữ nghĩa, đồ thị và từ khóa để có kết quả toàn diện nhất."
    )
    async def retrieve_comprehensive(
        self,
        query_text: str,
        entities: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện truy xuất lai và tổng hợp kết quả.

        Args:
            query_text (str): Chuỗi truy vấn gốc của người dùng.
            entities (List[str]): Danh sách các thực thể đã được trích xuất từ truy vấn (ví dụ: ['đàn guitar', 'bãi biển']).
            top_k (int): Số lượng kết quả cuối cùng mong muốn.

        Returns:
            List[Dict[str, Any]]: Danh sách kết quả cuối cùng đã được tổng hợp và xếp hạng.
        """
        # Bước 1: Mã hóa truy vấn văn bản thành vector
        query_vector = self.text_embedder.get_embedding_from_text(query_text)
        if query_vector is None:
            logger.warning("Không thể tạo vector cho truy vấn. Bỏ qua truy xuất.")
            return []
        query_vector_list = query_vector.flatten().tolist()

        # Bước 2: Chuẩn bị các tác vụ truy xuất song song
        # - Tìm kiếm ngữ nghĩa trên hình ảnh và văn bản OCR
        # - Tìm kiếm các cảnh chứa thực thể đã biết
        tasks = [
            self.vector_retriever.retrieve_by_vector(query_vector_list, 'visual_embedding', top_k),
            self.vector_retriever.retrieve_by_vector(query_vector_list, 'ocr_embedding', top_k)
        ]
        for entity in entities:
            tasks.append(self.graph_retriever.find_scenes_by_entity(entity, top_k))

        results_from_all_sources = await asyncio.gather(*tasks, return_exceptions=True)

        # Bước 3: Tổng hợp và xếp hạng kết quả bằng Reciprocal Rank Fusion
        valid_results = [res for res in results_from_all_sources if isinstance(res, list) and res]
        if not valid_results:
            logger.info("Không tìm thấy kết quả từ bất kỳ nguồn truy xuất nào.")
            return []
            
        fused_scores = self._reciprocal_rank_fusion(valid_results)
        sorted_scenes = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        # Bước 4: Lấy thông tin chi tiết cho top_k scene có điểm số cao nhất
        top_scene_ids = [scene_id for scene_id, score in sorted_scenes[:top_k]]
        final_results = self._get_scene_details(top_scene_ids)
        
        logger.info(f"Truy xuất lai hoàn tất. Trả về {len(final_results)} kết quả tổng hợp.")
        return final_results

    def _reciprocal_rank_fusion(self, search_results_lists: List[List[Dict]], k: int = 60) -> Dict[str, float]:
        """
        Thực hiện thuật toán Reciprocal Rank Fusion (RRF) để tổng hợp điểm số từ nhiều danh sách kết quả.
        
        Args:
            search_results_lists: Một danh sách chứa nhiều danh sách kết quả.
            k (int): Một hằng số để tránh chia cho 0.
            
        Returns:
            Một dictionary với scene_id là key và điểm số RRF tổng hợp là value.
        """
        fused_scores: Dict[str, float] = {}
        for result_list in search_results_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc.get('scene_id')
                if doc_id:
                    fused_scores.setdefault(doc_id, 0.0)
                    fused_scores[doc_id] += 1.0 / (k + rank + 1)
        return fused_scores

    def _get_scene_details(self, scene_ids: List[str]) -> List[Dict[str, Any]]:
        """Lấy thông tin chi tiết của các scene từ DB dựa trên danh sách ID."""
        if not scene_ids:
            return []
        try:
            tbl = self.vector_retriever._db_conn.open_table("scenes")
            # Tạo một filter SQL để lấy chính xác các scene_id
            id_filter = ", ".join([f"'{id_}'" for id_ in scene_ids])
            results = tbl.search().where(f"scene_id IN ({id_filter})").to_list()
            
            # Sắp xếp lại kết quả theo đúng thứ tự của scene_ids đã được xếp hạng
            results_map = {res['scene_id']: res for res in results}
            sorted_results = [results_map[id_] for id_ in scene_ids if id_ in results_map]
            return sorted_results
        except Exception as e:
            logger.error(f"Không thể lấy chi tiết scene: {e}", exc_info=True)
            return [{"scene_id": id_, "error": "Could not fetch details"} for id_ in scene_ids]

