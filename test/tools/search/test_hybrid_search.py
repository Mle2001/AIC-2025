# /tests/tools/search/test_hybrid_search.py

import pytest
from unittest.mock import MagicMock

# Nhập các công cụ cần kiểm thử và các công cụ phụ thuộc
from tools.search.hybrid_search import HybridSearchTool
from tools.search.vector_search import VectorSearchTool
from tools.search.graph_search import GraphSearchTool

# --- FIXTURES ---

@pytest.fixture
def mock_search_tools():
    """Fixture để tạo các mock object cho VectorSearchTool và GraphSearchTool."""
    mock_vector_tool = MagicMock(spec=VectorSearchTool)
    mock_graph_tool = MagicMock(spec=GraphSearchTool)
    return mock_vector_tool, mock_graph_tool

@pytest.fixture
def hybrid_tool(mock_search_tools) -> HybridSearchTool:
    """Cung cấp một instance của HybridSearchTool với các tool đã được mock."""
    return HybridSearchTool(vector_tool=mock_search_tools[0], graph_tool=mock_search_tools[1])

# --- TESTS ---

def test_multi_modal_search_calls_all_tools(hybrid_tool, mock_search_tools):
    """
    Kiểm thử tool 'multi_modal_search' gọi đến tất cả các công cụ con khi được yêu cầu.
    """
    # 1. ARRANGE
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    # Dữ liệu giả lập trả về từ các tool con
    mock_vector_tool.hybrid_search.return_value = [{"video_id": "vid1", "_distance": 0.1}]
    mock_graph_tool.search_graph.return_value = [{"video_id": "vid2", "name": "PersonA"}]
    
    query = "find PersonA in vid1"
    query_vector = [0.1] * 10
    entities = ["PersonA"]

    # 2. ACT
    hybrid_tool.multi_modal_search(
        query=query,
        query_vector=query_vector,
        entities=entities,
        search_types=["vector", "graph"]
    )

    # 3. ASSERT
    # Kiểm tra xem các tool con có được gọi với đúng tham số không
    mock_vector_tool.hybrid_search.assert_called_once_with(query_text=query, query_vector=query_vector)
    mock_graph_tool.search_graph.assert_called_once_with(entities=entities)

def test_multi_modal_search_merges_and_ranks_results(hybrid_tool, mock_search_tools):
    """
    Kiểm thử logic hợp nhất và xếp hạng kết quả.
    """
    # 1. ARRANGE
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    # vid1 xuất hiện ở cả 2 kết quả, nên điểm sẽ cao hơn
    mock_vector_tool.hybrid_search.return_value = [
        {"video_id": "vid1", "_distance": 0.1}, 
        {"video_id": "vid3", "_distance": 0.5}
    ]
    mock_graph_tool.search_graph.return_value = [
        {"video_id": "vid1", "name": "PersonA"},
        {"video_id": "vid2", "name": "LocationX"}
    ]
    
    # 2. ACT
    final_results = hybrid_tool.multi_modal_search(
        query="test",
        query_vector=[0.1],
        entities=["PersonA"],
        search_types=["vector", "graph"]
    )

    # 3. ASSERT
    assert len(final_results) == 3
    # vid1 phải đứng đầu vì có điểm cao nhất (từ cả vector và graph)
    assert final_results[0]["video_id"] == "vid1"
    # vid2 và vid3 sẽ xếp sau, thứ tự phụ thuộc vào điểm số đã tính
    # Điểm vid2 (graph) = 0.5
    # Điểm vid3 (vector) = 1 / (0.5 + 0.1) = 1.66...
    # Vậy vid3 sẽ đứng thứ 2
    assert final_results[1]["video_id"] == "vid3"
    assert final_results[2]["video_id"] == "vid2"

def test_multi_modal_search_handles_tool_errors_gracefully(hybrid_tool, mock_search_tools):
    """
    Kiểm thử trường hợp một trong các tool con bị lỗi, hệ thống vẫn hoạt động.
    """
    # 1. ARRANGE
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    # Giả lập VectorSearchTool bị lỗi, nhưng GraphSearchTool hoạt động bình thường
    mock_vector_tool.hybrid_search.side_effect = Exception("Vector DB is down")
    mock_graph_tool.search_graph.return_value = [{"video_id": "vid2"}]
    
    # 2. ACT
    final_results = hybrid_tool.multi_modal_search(
        query="test",
        query_vector=[0.1],
        entities=["PersonA"],
        search_types=["vector", "graph"]
    )
    
    # 3. ASSERT
    # Kết quả vẫn trả về từ GraphSearchTool
    assert len(final_results) == 1
    assert final_results[0]["video_id"] == "vid2"
    mock_vector_tool.hybrid_search.assert_called_once()
    mock_graph_tool.search_graph.assert_called_once()
