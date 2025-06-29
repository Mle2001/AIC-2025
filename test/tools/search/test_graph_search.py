# /tests/tools/search/test_graph_search.py

import pytest
from unittest.mock import patch, MagicMock

# Nhập công cụ cần kiểm thử
from tools.search.graph_search import GraphSearchTool

# --- FIXTURES ---

@pytest.fixture
def mock_neo4j_driver():
    """Fixture để mock thư viện neo4j driver."""
    with patch('tools.search.graph_search.GraphDatabase.driver') as mock_driver_cls:
        mock_driver_inst = MagicMock()
        mock_session_inst = MagicMock()
        mock_result_inst = MagicMock()

        # Cấu hình chuỗi mock
        mock_driver_cls.return_value = mock_driver_inst
        mock_driver_inst.session.return_value.__enter__.return_value = mock_session_inst
        mock_session_inst.run.return_value = mock_result_inst
        
        yield {
            "driver_cls": mock_driver_cls,
            "driver_inst": mock_driver_inst,
            "session_inst": mock_session_inst,
            "result_inst": mock_result_inst
        }

# --- TESTS ---

def test_graph_search_tool_success(mock_neo4j_driver):
    """
    Kiểm thử tool 'search_graph' trong trường hợp thành công.
    """
    # 1. ARRANGE
    tool = GraphSearchTool(uri="bolt://db", username="user", password="pw")
    
    # Dữ liệu giả lập trả về
    mock_node = MagicMock()
    mock_node._properties = {"name": "John Doe", "age": 30}
    mock_neo4j_driver["result_inst"].__iter__.return_value = [{"n": mock_node}]

    # 2. ACT
    results = tool.search_graph(entities=["Person"])

    # 3. ASSERT
    assert len(results) == 1
    assert results[0]["name"] == "John Doe"
    
    # Kiểm tra truy vấn Cypher đã được gọi
    mock_neo4j_driver["session_inst"].run.assert_called_once()
    # Lấy ra câu query đã được gọi
    called_query = mock_neo4j_driver["session_inst"].run.call_args[0][0]
    assert "WHERE any(label IN labels(n) WHERE label IN $entity_labels)" in called_query

def test_find_related_entities_tool_success(mock_neo4j_driver):
    """
    Kiểm thử tool 'find_related_entities' thành công.
    """
    # 1. ARRANGE
    tool = GraphSearchTool(uri="bolt://db", username="user", password="pw")
    
    mock_related_node = MagicMock()
    mock_related_node._properties = {"title": "The Matrix"}
    mock_neo4j_driver["result_inst"].__iter__.return_value = [{"relatedNode": mock_related_node}]

    # 2. ACT
    results = tool.find_related_entities(entity="Keanu Reeves", max_depth=2)

    # 3. ASSERT
    assert len(results) == 1
    assert results[0]["title"] == "The Matrix"
    
    # Kiểm tra truy vấn Cypher
    mock_neo4j_driver["session_inst"].run.assert_called_once()
    called_query = mock_neo4j_driver["session_inst"].run.call_args[0][0]
    assert "MATCH (startNode {name: $entity_name})-[*1..2]-(relatedNode)" in called_query
    called_params = mock_neo4j_driver["session_inst"].run.call_args[1]
    assert called_params["entity_name"] == "Keanu Reeves"

def test_init_handles_connection_error(mock_neo4j_driver):
    """
    Kiểm thử việc khởi tạo tool ném ra ConnectionError nếu không kết nối được DB.
    """
    # 1. ARRANGE
    mock_driver_inst = mock_neo4j_driver["driver_inst"]
    mock_driver_inst.verify_connectivity.side_effect = Exception("Authentication failed")

    # 2. ACT & 3. ASSERT
    with pytest.raises(ConnectionError, match="Không thể kết nối tới Neo4j"):
        GraphSearchTool(uri="bolt://db", username="user", password="wrong_pw")

def test_tool_raises_error_if_driver_not_initialized():
    """
    Kiểm thử tool sẽ báo lỗi nếu được sử dụng mà không có kết nối driver.
    """
    # 1. ARRANGE
    # Giả lập việc khởi tạo thất bại bằng cách patch __init__
    with patch.object(GraphSearchTool, "__init__", return_value=None):
        tool = GraphSearchTool(uri="", username="", password="")
        tool._driver = None # Đảm bảo driver là None

    # 2. ACT & 3. ASSERT
    with pytest.raises(ConnectionError, match="Driver Neo4j chưa được khởi tạo."):
        tool.search_graph(entities=["Person"])
