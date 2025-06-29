# /test/retrieval/test_graph_retriever.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from tools.retrieval.graph_retriever import GraphRetrieverTool

@pytest.fixture
def mock_neo4j_driver():
    """Fixture để mock thư viện neo4j driver bất đồng bộ."""
    with patch('tools.retrieval.graph_retriever.AsyncGraphDatabase.driver') as mock_driver_cls:
        mock_driver_inst = MagicMock()
        mock_driver_inst.verify_connectivity = AsyncMock()
        mock_driver_inst.close = AsyncMock()
        
        mock_session_inst = MagicMock()
        mock_session_inst.__aenter__ = AsyncMock(return_value=mock_session_inst)
        mock_session_inst.__aexit__ = AsyncMock()
        mock_session_inst.run = AsyncMock()
        
        mock_driver_inst.session.return_value = mock_session_inst
        mock_driver_cls.return_value = mock_driver_inst
        
        yield {
            "driver_cls": mock_driver_cls,
            "driver_inst": mock_driver_inst,
            "session_inst": mock_session_inst,
        }

@pytest.mark.asyncio
async def test_retrieve_entities_success(mock_neo4j_driver):
    """Kiểm thử tool 'retrieve_entities_by_property' thành công."""
    tool = GraphRetrieverTool(uri="bolt://db", username="user", password="pw")
    
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[{"n": {"name": "John Doe", "age": 30}}])
    mock_neo4j_driver["session_inst"].run.return_value = mock_result
    
    properties = {"name": "John Doe"}
    results = await tool.retrieve_entities_by_property(entity_label="Person", properties=properties)
    
    assert len(results) == 1
    assert results[0]["name"] == "John Doe"
    await tool.close()

@pytest.mark.asyncio
async def test_find_related_entities_success(mock_neo4j_driver):
    """Kiểm thử tool 'find_related_entities' thành công."""
    tool = GraphRetrieverTool(uri="bolt://db", username="user", password="pw")
    
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[{"relatedNode": {"title": "The Matrix"}}])
    mock_neo4j_driver["session_inst"].run.return_value = mock_result

    results = await tool.find_related_entities(entity_name="Keanu Reeves", entity_label="Actor", max_depth=2)

    assert len(results) == 1
    assert results[0]["title"] == "The Matrix"
    await tool.close()

@pytest.mark.asyncio
async def test_verify_connectivity_handles_exception(mock_neo4j_driver):
    """
    ✅ SỬA LỖI: Kiểm tra hàm _verify_connectivity thay vì __init__.
    Kiểm thử việc xác minh kết nối ném ra ConnectionError khi có lỗi.
    """
    # 1. ARRANGE
    tool = GraphRetrieverTool(uri="bolt://db", username="user", password="pw")
    mock_driver_inst = mock_neo4j_driver["driver_inst"]
    mock_driver_inst.verify_connectivity.side_effect = Exception("Authentication failed")

    # 2. ACT & 3. ASSERT
    with pytest.raises(ConnectionError, match="Không thể xác minh kết nối Neo4j"):
        await tool._verify_connectivity()

@pytest.mark.asyncio
async def test_tool_raises_error_if_driver_not_initialized():
    """Kiểm thử tool sẽ báo lỗi nếu được sử dụng mà không có kết nối driver."""
    with patch.object(GraphRetrieverTool, "__init__", return_value=None):
        tool = GraphRetrieverTool(uri="", username="", password="")
        tool._driver = None

    with pytest.raises(ConnectionError, match="Driver Neo4j chưa được khởi tạo."):
        await tool.retrieve_entities_by_property(entity_label="Person", properties={})
