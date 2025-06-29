# /test/retrieval/test_vector_retriever.py
import pytest
from unittest.mock import patch, MagicMock

from tools.retrieval.vector_retriever import VectorRetrieverTool

@pytest.fixture
def mock_lancedb():
    """Fixture để mock thư viện lancedb."""
    with patch('tools.retrieval.vector_retriever.lancedb') as mock_lancedb_lib:
        mock_db_conn = MagicMock()
        mock_lancedb_lib.connect.return_value = mock_db_conn
        
        mock_table = MagicMock()
        mock_db_conn.open_table.return_value = mock_table
        
        mock_search_result = MagicMock()
        mock_table.search.return_value = mock_search_result
        mock_search_result.limit.return_value = mock_search_result
        mock_search_result.query.return_value = mock_search_result
        
        yield {
            "lib": mock_lancedb_lib,
            "conn": mock_db_conn,
            "table": mock_table,
            "search_result": mock_search_result
        }

def test_retrieve_by_vector_success(mock_lancedb):
    """Kiểm thử tool 'retrieve_by_vector' thành công."""
    tool = VectorRetrieverTool()
    query_vector = [0.1] * 128
    expected_results = [{"id": 1, "vector": [], "_distance": 0.1}]
    
    mock_lancedb["search_result"].to_list.return_value = expected_results

    results = tool.retrieve_by_vector(query_vector=query_vector, table_name="test_table", limit=5)

    assert results == expected_results
    mock_lancedb["table"].search.assert_called_once_with(query_vector)
    mock_lancedb["search_result"].limit.assert_called_once_with(5)

def test_hybrid_retrieval_success(mock_lancedb):
    """Kiểm thử tool 'hybrid_retrieval' thành công."""
    tool = VectorRetrieverTool()
    query_text = "a cat on a roof"
    query_vector = [0.2] * 128
    expected_results = [{"id": 2, "text": "a cat on a roof", "_distance": 0.2}]
    
    mock_lancedb["search_result"].to_list.return_value = expected_results

    results = tool.hybrid_retrieval(
        query_text=query_text,
        query_vector=query_vector,
        table_name="hybrid_table",
        limit=3
    )

    assert results == expected_results
    mock_lancedb["table"].search.assert_called_once_with(query_vector, query_type="hybrid")
    mock_lancedb["search_result"].query.assert_called_once_with(query_text)
    mock_lancedb["search_result"].limit.assert_called_once_with(3)

def test_retrieval_tool_handles_db_exception(mock_lancedb):
    """Kiểm thử tool ném ra ngoại lệ khi DB lỗi."""
    tool = VectorRetrieverTool()
    mock_lancedb["table"].search.side_effect = Exception("Table does not exist")

    with pytest.raises(Exception, match="Table does not exist"):
        tool.retrieve_by_vector(query_vector=[0.1], table_name="non_existent_table")

def test_init_handles_connection_error(mock_lancedb):
    """Kiểm thử việc khởi tạo ném ra ConnectionError nếu không kết nối được DB."""
    mock_lancedb["lib"].connect.side_effect = ConnectionError("Failed to connect")
    with pytest.raises(ConnectionError, match="Failed to connect"):
        VectorRetrieverTool()
