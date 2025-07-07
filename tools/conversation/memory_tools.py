# /tools/conversation/memory_tools.py
"""
Công cụ để quản lý bộ nhớ dài hạn và sở thích của người dùng.
"""
import logging
from typing import Dict, Any

from agno.tools import tool

# Giả định có kết nối tới DB người dùng
# from database.connections import get_user_profile_db_connection

logger = logging.getLogger(__name__)

class MemoryTool:
    """
    Công cụ giúp Agent ghi nhớ và truy xuất thông tin về người dùng qua các phiên.
    """

    def __init__(self):
        # self.user_db = get_user_profile_db_connection()
        logger.info("MemoryTool đã được khởi tạo.")

    @tool(
        name="store_user_preference",
        description="Lưu một thông tin hoặc sở thích của người dùng vào bộ nhớ dài hạn (ví dụ: 'người dùng thích video về mèo')."
    )
    def store_user_preference(self, user_id: str, preference_key: str, preference_value: Any) -> bool:
        """
        Lưu một sở thích của người dùng.

        Args:
            user_id (str): ID định danh của người dùng.
            preference_key (str): Tên của sở thích (ví dụ: 'favorite_topic').
            preference_value (Any): Giá trị của sở thích (ví dụ: 'khoa học vũ trụ').

        Returns:
            bool: True nếu lưu thành công, False nếu thất bại.
        """
        logger.info(f"Đang lưu sở thích '{preference_key}'='{preference_value}' cho user_id: {user_id}")
        # --- LOGIC THỰC TẾ ---
        # Code để cập nhật hồ sơ người dùng trong PostgreSQL hoặc một DB tương tự.
        # Ví dụ:
        # success = self.user_db.update_profile(
        #     user_id, {'preferences': {preference_key: preference_value}}
        # )
        # return success

        # --- GIẢ LẬP ---
        return True

    @tool(
        name="get_user_preferences",
        description="Lấy thông tin và các sở thích đã biết về người dùng để cá nhân hóa câu trả lời."
    )
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Lấy hồ sơ sở thích của người dùng.

        Args:
            user_id (str): ID định danh của người dùng.

        Returns:
            Dict[str, Any]: Một dictionary chứa các sở thích đã lưu của người dùng.
        """
        logger.info(f"Đang lấy sở thích cho user_id: {user_id}")
        # --- LOGIC THỰC TẾ ---
        # Code để truy vấn hồ sơ người dùng từ DB.
        # Ví dụ:
        # profile = self.user_db.get_profile(user_id)
        # return profile.get('preferences', {})

        # --- DỮ LIỆU GIẢ LẬP ĐỂ TEST ---
        mock_preferences = {
            "favorite_topic": "khoa học vũ trụ",
            "preferred_language": "tiếng Việt",
            "interaction_style": "chi tiết, học thuật"
        }
        return mock_preferences
