# /test/retrieval/test_hybrid_retriever.py
import pytest
from unittest.mock import MagicMock, AsyncMock

from tools.retrieval.hybrid_retriever import HybridRetrieverTool
from tools.retrieval.vector_retriever import VectorRetrieverTool
from tools.retrieval.graph_retriever import GraphRetrieverTool

@pytest.fixture
def mock_search_tools():
    """Fixture để tạo các mock object cho các tool con."""
    mock_vector_tool = MagicMock(spec=VectorRetrieverTool)
    mock_vector_tool.hybrid_retrieval = AsyncMock()
    
    mock_graph_tool = MagicMock(spec=GraphRetrieverTool)
    mock_graph_tool.retrieve_entities_by_property = AsyncMock()
    
    return mock_vector_tool, mock_graph_tool

@pytest.fixture
def hybrid_tool(mock_search_tools) -> HybridRetrieverTool:
    """Cung cấp một instance của HybridRetrieverTool với các tool đã được mock."""
    return HybridRetrieverTool(vector_tool=mock_search_tools[0], graph_tool=mock_search_tools[1])

@pytest.mark.asyncio
async def test_multi_modal_retrieval_calls_all_tools(hybrid_tool, mock_search_tools):
    """Kiểm thử tool gọi đến tất cả các công cụ con khi được yêu cầu."""
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    mock_vector_tool.hybrid_retrieval.return_value = []
    mock_graph_tool.retrieve_entities_by_property.return_value = []
    
    query = "find PersonA in vid1"
    query_vector = [0.1] * 10
    entities = [{"label": "Person", "name": "PersonA"}]

    await hybrid_tool.multi_modal_retrieval(
        query_text=query,
        query_vector=query_vector,
        entities=entities,
        search_types=["vector", "graph"]
    )

    mock_vector_tool.hybrid_retrieval.assert_called_once_with(query_text=query, query_vector=query_vector)
    mock_graph_tool.retrieve_entities_by_property.assert_called_once_with(
        entity_label="Person", properties={"name": "PersonA"}
    )

@pytest.mark.asyncio
async def test_multi_modal_retrieval_merges_and_ranks_results(hybrid_tool, mock_search_tools):
    """Kiểm thử logic hợp nhất và xếp hạng kết quả."""
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    mock_vector_tool.hybrid_retrieval.return_value = [
        {"video_id": "vid1", "_distance": 0.1}, 
        {"video_id": "vid3", "_distance": 0.5}
    ]
    mock_graph_tool.retrieve_entities_by_property.return_value = [
        {"video_id": "vid1", "name": "PersonA"},
        {"video_id": "vid2", "name": "LocationX"}
    ]
    
    final_results = await hybrid_tool.multi_modal_retrieval(
        query_text="test",
        query_vector=[0.1],
        entities=[{"label": "Person", "name": "PersonA"}],
        search_types=["vector", "graph"]
    )

    assert len(final_results) == 3
    # vid1 phải đứng đầu vì có điểm cao nhất (từ cả vector và graph)
    assert final_results[0]["video_id"] == "vid1"
    # Điểm vid3 (vector) = 1 + 1 / (0.5 + 0.1) = 2.66...
    # Điểm vid2 (graph) = 0.8
    # Vậy vid3 sẽ đứng thứ 2
    assert final_results[1]["video_id"] == "vid3"
    assert final_results[2]["video_id"] == "vid2"

@pytest.mark.asyncio
async def test_multi_modal_retrieval_handles_tool_errors_gracefully(hybrid_tool, mock_search_tools):
    """Kiểm thử trường hợp một trong các tool con bị lỗi, hệ thống vẫn hoạt động."""
    mock_vector_tool, mock_graph_tool = mock_search_tools
    
    mock_vector_tool.hybrid_retrieval.side_effect = Exception("Vector DB is down")
    mock_graph_tool.retrieve_entities_by_property.return_value = [{"video_id": "vid2"}]
    
    final_results = await hybrid_tool.multi_modal_retrieval(
        query_text="test",
        query_vector=[0.1],
        entities=[{"label": "Person", "name": "PersonA"}],
        search_types=["vector", "graph"]
    )
    
    # Kết quả vẫn trả về từ GraphRetrieverTool
    assert len(final_results) == 1
    assert final_results[0]["video_id"] == "vid2"
    mock_vector_tool.hybrid_retrieval.assert_called_once()
    mock_graph_tool.retrieve_entities_by_property.assert_called_once()
