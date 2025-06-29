# /tests/tools/conversation/test_context_tools.py

import pytest
from unittest.mock import MagicMock, AsyncMock

# Nhập công cụ cần kiểm thử và các lớp phụ thuộc
from tools.conversation.context_tools import ContextTool
from database.connections.cache_db import CacheDB

# --- FIXTURES ---

@pytest.fixture
def mock_cache_db():
    """Fixture để tạo một mock object cho CacheDB."""
    # Sử dụng AsyncMock vì các phương thức của CacheDB là async
    mock_db = MagicMock(spec=CacheDB)
    mock_db.get_chat_history = AsyncMock()
    return mock_db

@pytest.fixture
def context_tool(mock_cache_db) -> ContextTool:
    """Cung cấp một instance của ContextTool với CacheDB đã được mock."""
    return ContextTool(cache_db=mock_cache_db)

# --- TESTS ---

@pytest.mark.asyncio
async def test_get_conversation_history_success(context_tool, mock_cache_db):
    """
    Kiểm thử tool 'get_conversation_history' trong trường hợp thành công.
    """
    # 1. ARRANGE
    session_id = "session-123"
    num_turns = 3
    # Dữ liệu giả lập trả về từ cache (thứ tự ngược)
    mock_history = [
        {"role": "assistant", "message": "Đây là video về mèo."},
        {"role": "user", "message": "Tìm video về mèo."},
    ]
    mock_cache_db.get_chat_history.return_value = mock_history

    # 2. ACT
    result = await context_tool.get_conversation_history(session_id=session_id, num_turns=num_turns)

    # 3. ASSERT
    assert len(result) == 2
    # Kiểm tra rằng lịch sử đã được đảo ngược lại đúng thứ tự
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    
    # Kiểm tra lời gọi đến cache
    mock_cache_db.get_chat_history.assert_called_once_with(session_id, limit=num_turns)

@pytest.mark.asyncio
async def test_get_conversation_history_handles_db_error(context_tool, mock_cache_db):
    """
    Kiểm thử tool ném ra ngoại lệ khi cache bị lỗi.
    """
    # 1. ARRANGE
    mock_cache_db.get_chat_history.side_effect = ConnectionError("Redis is down")

    # 2. ACT & 3. ASSERT
    with pytest.raises(ConnectionError, match="Redis is down"):
        await context_tool.get_conversation_history(session_id="session-error")

def test_extract_context_entities_success(context_tool):
    """
    Kiểm thử logic trích xuất thực thể đơn giản từ lịch sử hội thoại.
    """
    # 1. ARRANGE
    history = [
        {"role": "user", "message": "Hãy tìm cho tôi thông tin về Hà Nội."},
        {"role": "assistant", "message": "Hà Nội là thủ đô của Việt Nam."},
        {"role": "user", "message": "Thời tiết ở đó thế nào?"} # "đó" ở đây chính là "Hà Nội"
    ]

    # 2. ACT
    entities = context_tool.extract_context_entities(conversation_history=history)

    # 3. ASSERT
    assert len(entities) == 1
    assert entities[0]["name"] == "Hà Nội"
    assert entities[0]["type"] == "PROPER_NOUN"

def test_extract_context_entities_empty_history(context_tool):
    """
    Kiểm thử tool hoạt động đúng với lịch sử rỗng.
    """
    # 1. ARRANGE
    history = []
    # 2. ACT
    entities = context_tool.extract_context_entities(conversation_history=history)
    # 3. ASSERT
    assert entities == []
