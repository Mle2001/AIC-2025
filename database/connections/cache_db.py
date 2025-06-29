# /database/connections/cache_db.py
"""
Redis connection cho caching và session storage.
"""
import redis.asyncio as redis
import logging
import json
from typing import Dict, Any, Optional, List

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheDB:
    """
    Lớp quản lý kết nối và các thao tác với Redis cho việc caching
    và lưu trữ dữ liệu tạm thời như lịch sử hội thoại.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Khởi tạo kết nối tới Redis.

        Args:
            redis_url (str): URL kết nối của Redis.
        """
        try:
            # from_url sẽ tự động quản lý connection pool
            self.redis = redis.from_url(redis_url, decode_responses=True)
            logger.info(f"Đã tạo pool kết nối tới Redis tại: {redis_url}")
        except Exception as e:
            logger.error(f"Không thể kết nối tới Redis: {e}", exc_info=True)
            self.redis = None

    async def verify_connection(self) -> bool:
        """Kiểm tra kết nối tới Redis."""
        if not self.redis:
            return False
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử hội thoại từ Redis (dạng list).

        Args:
            session_id (str): ID của phiên hội thoại.
            limit (int): Số lượng tin nhắn tối đa cần lấy.

        Returns:
            List[Dict[str, Any]]: Danh sách các tin nhắn.
        """
        if not self.redis:
            raise ConnectionError("Kết nối Redis chưa được thiết lập.")
        
        history_key = f"history:{session_id}"
        try:
            # Lấy ra các phần tử từ 0 đến limit-1
            history_json = await self.redis.lrange(history_key, 0, limit - 1)
            # Mỗi phần tử là một chuỗi JSON, cần parse lại
            return [json.loads(msg) for msg in history_json]
        except Exception as e:
            logger.error(f"Lỗi khi lấy lịch sử chat cho session '{session_id}': {e}")
            return []

    async def add_to_chat_history(self, session_id: str, message: Dict[str, Any]):
        """
        Thêm một tin nhắn mới vào đầu danh sách lịch sử hội thoại.

        Args:
            session_id (str): ID của phiên hội thoại.
            message (Dict[str, Any]): Nội dung tin nhắn cần lưu.
        """
        if not self.redis:
            raise ConnectionError("Kết nối Redis chưa được thiết lập.")
            
        history_key = f"history:{session_id}"
        try:
            # Chuyển dict thành chuỗi JSON để lưu
            message_json = json.dumps(message)
            # Thêm vào đầu list
            await self.redis.lpush(history_key, message_json)
            # Giới hạn độ dài của list để không bị đầy bộ nhớ
            await self.redis.ltrim(history_key, 0, 99)
            # Đặt thời gian hết hạn cho key để tự động dọn dẹp
            await self.redis.expire(history_key, 86400) # 24 giờ
        except Exception as e:
            logger.error(f"Lỗi khi thêm vào lịch sử chat cho session '{session_id}': {e}")

    async def clear_session(self, session_id: str):
        """Xóa các key liên quan đến một session."""
        if not self.redis:
            raise ConnectionError("Kết nối Redis chưa được thiết lập.")
            
        keys_to_delete = [f"history:{session_id}", f"context:{session_id}"]
        try:
            # Xóa nhiều key cùng lúc
            await self.redis.delete(*keys_to_delete)
            logger.info(f"Đã xóa dữ liệu cho session: {session_id}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa session '{session_id}': {e}")

    async def close(self):
        """Đóng kết nối Redis."""
        if self.redis:
            await self.redis.close()
