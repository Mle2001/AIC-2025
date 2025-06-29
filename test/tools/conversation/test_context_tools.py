# /test/tools/conversation/test_context_tools.py
import pytest
from unittest.mock import MagicMock, AsyncMock

from tools.conversation.context_tools import ContextTool
from database.connections.cache_db import CacheDB

@pytest.fixture
def mock_cache_db():
    """Fixture để tạo một mock object cho CacheDB."""
    mock_db = MagicMock(spec=CacheDB)
    mock_db.get_chat_history = AsyncMock()
    return mock_db

@pytest.fixture
def context_tool(mock_cache_db) -> ContextTool:
    """Cung cấp một instance của ContextTool với CacheDB đã được mock."""
    return ContextTool(cache_db=mock_cache_db)

@pytest.mark.asyncio
async def test_get_conversation_history_success(context_tool, mock_cache_db):
    """Kiểm thử tool 'get_conversation_history' trong trường hợp thành công."""
    session_id = "session-123"
    num_turns = 3
    mock_history = [
        {"role": "assistant", "message": "Đây là video về mèo."},
        {"role": "user", "message": "Tìm video về mèo."},
    ]
    mock_cache_db.get_chat_history.return_value = mock_history

    result = await context_tool.get_conversation_history(session_id=session_id, num_turns=num_turns)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    mock_cache_db.get_chat_history.assert_called_once_with(session_id, limit=num_turns)

@pytest.mark.asyncio
async def test_get_conversation_history_handles_db_error(context_tool, mock_cache_db):
    """Kiểm thử tool ném ra ngoại lệ khi cache bị lỗi."""
    mock_cache_db.get_chat_history.side_effect = ConnectionError("Redis is down")
    with pytest.raises(ConnectionError, match="Redis is down"):
        await context_tool.get_conversation_history(session_id="session-error")

def test_extract_context_entities_success(context_tool):
    """Kiểm thử logic trích xuất thực thể từ lịch sử hội thoại."""
    history = [
        {"role": "user", "message": "Hãy tìm cho tôi thông tin về Hà Nội và 'Tháp Rùa'."},
        {"role": "assistant", "message": "Hà Nội là thủ đô của Việt Nam."}, # "Việt Nam" sẽ bị bỏ qua
        {"role": "user", "message": "Thời tiết ở đó thế nào? Tôi cũng muốn biết về Sài Gòn."}
    ]

    entities = context_tool.extract_context_entities(conversation_history=history)

    # ✅ SỬA LỖI: Kỳ vọng đúng là 3 thực thể từ các tin nhắn của người dùng.
    assert len(entities) == 3
    entity_names = {e['name'] for e in entities}
    assert "Hà Nội" in entity_names
    assert "Tháp Rùa" in entity_names
    assert "Sài Gòn" in entity_names
    assert "Việt Nam" not in entity_names # Đảm bảo "Việt Nam" không được trích xuất

def test_extract_context_entities_empty_history(context_tool):
    """Kiểm thử tool hoạt động đúng với lịch sử rỗng."""
    entities = context_tool.extract_context_entities(conversation_history=[])
    assert entities == []
