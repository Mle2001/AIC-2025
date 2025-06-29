# /tests/tools/search/test_vector_search.py

import pytest
from unittest.mock import patch, MagicMock

# Nhập công cụ cần kiểm thử
from tools.search.vector_search import VectorSearchTool

# --- FIXTURES ---

@pytest.fixture
def mock_lancedb():
    """Fixture để mock thư viện lancedb."""
    with patch('tools.search.vector_search.lancedb') as mock_lancedb_lib:
        # Mock connection object
        mock_db_conn = MagicMock()
        mock_lancedb_lib.connect.return_value = mock_db_conn
        
        # Mock table object
        mock_table = MagicMock()
        mock_db_conn.open_table.return_value = mock_table
        
        # Mock search result object
        mock_search_result = MagicMock()
        mock_table.search.return_value = mock_search_result
        mock_search_result.limit.return_value = mock_search_result
        mock_search_result.query.return_value = mock_search_result # For hybrid search chaining
        
        yield {
            "lib": mock_lancedb_lib,
            "conn": mock_db_conn,
            "table": mock_table,
            "search_result": mock_search_result
        }

# --- TESTS ---

def test_vector_search_tool_success(mock_lancedb):
    """
    Kiểm thử tool 'search_vectors' trong trường hợp thành công.
    """
    # 1. ARRANGE
    tool = VectorSearchTool()
    query_vector = [0.1] * 128 # Vector truy vấn giả
    expected_results = [{"id": 1, "vector": [], "_distance": 0.1}]
    
    # Cấu hình mock search result để trả về kết quả mong muốn
    mock_lancedb["search_result"].to_list.return_value = expected_results

    # 2. ACT
    results = tool.search_vectors(query_vector=query_vector, table_name="test_table", limit=5)

    # 3. ASSERT
    assert results == expected_results
    
    # Kiểm tra các lời gọi mock
    mock_lancedb["lib"].connect.assert_called_once()
    mock_lancedb["conn"].open_table.assert_called_once_with("test_table")
    mock_lancedb["table"].search.assert_called_once_with(query_vector)
    mock_lancedb["search_result"].limit.assert_called_once_with(5)
    mock_lancedb["search_result"].to_list.assert_called_once()

def test_hybrid_search_tool_success(mock_lancedb):
    """
    Kiểm thử tool 'hybrid_search' trong trường hợp thành công.
    """
    # 1. ARRANGE
    tool = VectorSearchTool()
    query_text = "a cat on a roof"
    query_vector = [0.2] * 128
    expected_results = [{"id": 2, "text": "a cat on a roof", "_distance": 0.2}]
    
    mock_lancedb["search_result"].to_list.return_value = expected_results

    # 2. ACT
    results = tool.hybrid_search(
        query_text=query_text,
        query_vector=query_vector,
        table_name="hybrid_table",
        limit=3
    )

    # 3. ASSERT
    assert results == expected_results
    
    # Kiểm tra chuỗi lời gọi cho hybrid search
    mock_lancedb["table"].search.assert_called_once_with(query_vector, query_type="hybrid")
    mock_lancedb["search_result"].query.assert_called_once_with(query_text)
    mock_lancedb["search_result"].limit.assert_called_once_with(3)

def test_search_tool_handles_db_exception(mock_lancedb):
    """
    Kiểm thử tool ném ra ngoại lệ khi cơ sở dữ liệu gặp lỗi.
    """
    # 1. ARRANGE
    tool = VectorSearchTool()
    mock_lancedb["table"].search.side_effect = Exception("Table does not exist")

    # 2. ACT & 3. ASSERT
    with pytest.raises(Exception, match="Table does not exist"):
        tool.search_vectors(query_vector=[0.1], table_name="non_existent_table")

def test_init_handles_connection_error(mock_lancedb):
    """
    Kiểm thử việc khởi tạo tool ném ra ConnectionError nếu không kết nối được DB.
    """
    # 1. ARRANGE
    mock_lancedb["lib"].connect.side_effect = ConnectionError("Failed to connect")

    # 2. ACT & 3. ASSERT
    with pytest.raises(ConnectionError, match="Failed to connect"):
        VectorSearchTool()
