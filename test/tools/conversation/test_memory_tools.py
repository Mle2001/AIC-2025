# /test/tools/conversation/test_memory_tools.py
import pytest
from unittest.mock import MagicMock, AsyncMock

from tools.conversation.memory_tools import MemoryTool
from database.connections.cache_db import CacheDB
from database.connections.metadata_db import MetadataDB

@pytest.fixture
def mock_dbs():
    """Fixture để tạo các mock object cho CacheDB và MetadataDB."""
    mock_cache = MagicMock(spec=CacheDB)
    mock_cache.redis = MagicMock()
    mock_cache.redis.hset = AsyncMock()
    mock_cache.redis.hgetall = AsyncMock()
    
    mock_metadata = MagicMock(spec=MetadataDB)
    # Giả lập phương thức async nếu có
    mock_metadata.update_user_preferences = AsyncMock()
    
    return mock_cache, mock_metadata

@pytest.fixture
def memory_tool(mock_dbs) -> MemoryTool:
    """Cung cấp một instance của MemoryTool với các DB đã được mock."""
    return MemoryTool(cache_db=mock_dbs[0], metadata_db=mock_dbs[1])

@pytest.mark.asyncio
async def test_store_user_preference_success(memory_tool, mock_dbs):
    """Kiểm thử tool 'store_user_preference' trong trường hợp thành công."""
    mock_cache, _ = mock_dbs
    user_id = "user-007"
    pref_type = "favorite_topic"
    pref_value = "science_fiction"
    
    mock_cache.redis.hset.return_value = 1

    result = await memory_tool.store_user_preference(
        user_id=user_id,
        preference_type=pref_type,
        preference_value=pref_value
    )

    assert result is True
    expected_key = f"user:{user_id}:preferences"
    mock_cache.redis.hset.assert_called_once_with(expected_key, pref_type, str(pref_value))

@pytest.mark.asyncio
async def test_get_user_preferences_success(memory_tool, mock_dbs):
    """Kiểm thử tool 'get_user_preferences' trong trường hợp thành công."""
    mock_cache, _ = mock_dbs
    user_id = "user-008"
    
    # Dữ liệu giả lập trả về từ Redis (đã được decode)
    mock_redis_data = {
        'favorite_topic': 'history',
        'language': 'vietnamese'
    }
    mock_cache.redis.hgetall.return_value = mock_redis_data

    result = await memory_tool.get_user_preferences(user_id=user_id)

    assert result == mock_redis_data
    expected_key = f"user:{user_id}:preferences"
    mock_cache.redis.hgetall.assert_called_once_with(expected_key)

@pytest.mark.asyncio
async def test_store_user_preference_requires_userid(memory_tool):
    """Kiểm thử tool ném ra ValueError nếu user_id bị thiếu."""
    with pytest.raises(ValueError, match="user_id không được để trống."):
        await memory_tool.store_user_preference(user_id="", preference_type="test", preference_value="test")

@pytest.mark.asyncio
async def test_get_user_preferences_empty(memory_tool, mock_dbs):
    """Kiểm thử tool trả về dict rỗng nếu người dùng chưa có sở thích nào."""
    mock_cache, _ = mock_dbs
    mock_cache.redis.hgetall.return_value = {}

    result = await memory_tool.get_user_preferences(user_id="new_user")

    assert result == {}
