# /tools/conversation/memory_tools.py
"""
Tool để quản lý bộ nhớ dài hạn của người dùng, tuân thủ kiến trúc Agno.
"""
import logging
from typing import Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Giả định có các lớp để tương tác với DB
# Chúng sẽ được mock trong các bài test.
from database.connections.cache_db import CacheDB
from database.connections.metadata_db import MetadataDB

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTool:
    """
    Một class chứa các công cụ để lưu trữ và truy xuất thông tin dài hạn
    về người dùng, như sở thích hoặc các thông tin đã biết.
    """
    def __init__(self, cache_db: CacheDB, metadata_db: MetadataDB):
        """
        Khởi tạo tool với các kết nối đến cơ sở dữ liệu cần thiết.
        Sử dụng Dependency Injection để dễ dàng kiểm thử.

        Args:
            cache_db (CacheDB): Instance của lớp quản lý cache (Redis).
            metadata_db (MetadataDB): Instance của lớp quản lý metadata (Postgres).
        """
        self.cache = cache_db
        self.db = metadata_db
        logger.info("MemoryTool đã được khởi tạo với CacheDB và MetadataDB.")

    @tool(
        name="store_user_preference",
        description="Lưu một sở thích hoặc thông tin của người dùng vào bộ nhớ dài hạn.",
        cache_results=False # Ghi dữ liệu, không cần cache
    )
    async def store_user_preference(self, user_id: str, preference_type: str, preference_value: Any) -> bool:
        """
        Lưu một cặp (key, value) sở thích của người dùng.

        Args:
            user_id (str): ID của người dùng.
            preference_type (str): Loại sở thích (ví dụ: 'favorite_topic', 'language').
            preference_value (Any): Giá trị của sở thích.

        Returns:
            bool: True nếu lưu thành công, False nếu ngược lại.
        """
        if not user_id:
            raise ValueError("user_id không được để trống.")
            
        logger.info(f"Đang lưu sở thích '{preference_type}' cho người dùng: {user_id}")
        try:
            # Dữ liệu có thể được lưu vào cả Redis (để truy cập nhanh) và Postgres (để lưu trữ lâu dài).
            # Ví dụ: Lưu vào Redis cache
            cache_key = f"user:{user_id}:preferences"
            await self.cache.redis.hset(cache_key, preference_type, str(preference_value))
            
            # Logic để cập nhật vào Postgres sẽ phức tạp hơn, có thể là cập nhật một trường JSON
            # await self.db.update_user_preferences(user_id, {preference_type: preference_value})
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu sở thích cho người dùng '{user_id}': {e}", exc_info=True)
            raise e

    @tool(
        name="get_user_preferences",
        description="Lấy tất cả các sở thích đã được lưu của một người dùng.",
        cache_results=True,
        cache_ttl=3600 # Cache sở thích người dùng trong 1 giờ
    )
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Truy xuất tất cả các sở thích đã lưu của một người dùng từ cache.

        Args:
            user_id (str): ID của người dùng.

        Returns:
            Dict[str, Any]: Một dictionary chứa tất cả các sở thích của người dùng.
        """
        if not user_id:
            raise ValueError("user_id không được để trống.")

        logger.info(f"Đang lấy sở thích cho người dùng: {user_id}")
        try:
            cache_key = f"user:{user_id}:preferences"
            preferences = await self.cache.redis.hgetall(cache_key)
            
            # Chuyển đổi giá trị từ bytes sang string
            decoded_preferences = {k.decode('utf-8'): v.decode('utf-8') for k, v in preferences.items()}
            
            return decoded_preferences
        except Exception as e:
            logger.error(f"Lỗi khi lấy sở thích của người dùng '{user_id}': {e}", exc_info=True)
            raise e
