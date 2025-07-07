# /tools/conversation/context_tools.py
"""
Công cụ để quản lý và phân tích ngữ cảnh của cuộc hội thoại.
"""
import logging
from typing import Dict, List, Any

from agno.tools import tool

# Giả định rằng chúng ta có một module để kết nối tới DB,
# nơi lưu trữ lịch sử hội thoại.
# from database.connections import get_conversation_db_connection

logger = logging.getLogger(__name__)

class ContextTool:
    """
    Công cụ giúp các Agent truy xuất và hiểu ngữ cảnh từ các lượt hội thoại trước.
    """

    def __init__(self):
        # self.db_conn = get_conversation_db_connection()
        logger.info("ContextTool đã được khởi tạo.")

    @tool(
        name="get_conversation_history",
        description="Lấy lịch sử của cuộc hội thoại dựa trên ID của phiên (session). Rất quan trọng để hiểu các câu hỏi nối tiếp.",
        cache_results=False # Lịch sử hội thoại luôn cần được cập nhật mới nhất
    )
    def get_conversation_history(self, session_id: str, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Lấy N lượt hội thoại cuối cùng của một phiên.

        Args:
            session_id (str): ID định danh cho phiên hội thoại.
            num_turns (int): Số lượng lượt hội thoại gần nhất cần lấy.

        Returns:
            List[Dict[str, Any]]: Danh sách các lượt hội thoại, mỗi lượt là một dict
                                  chứa {'role': 'user'|'assistant', 'content': '...'}
        """
        logger.info(f"Đang truy xuất {num_turns} lượt hội thoại cuối cho session_id: {session_id}")
        # --- LOGIC THỰC TẾ ---
        # Tại đây sẽ là code để truy vấn vào Redis hoặc PostgreSQL
        # để lấy lịch sử hội thoại đã được lưu.
        # Ví dụ:
        # history = self.db_conn.query(
        #     "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        #     (session_id, num_turns)
        # )
        # return history

        # --- DỮ LIỆU GIẢ LẬP ĐỂ TEST ---
        mock_history = [
            {"role": "user", "content": "Tìm cho tôi các video về nấu ăn."},
            {"role": "assistant", "content": "Tôi tìm thấy 5 video về nấu ăn. Bạn có muốn xem video về món Âu hay Á?"},
            {"role": "user", "content": "Món Âu đi."},
        ]
        return mock_history[-num_turns:] if num_turns < len(mock_history) else mock_history

    @tool(
        name="summarize_context",
        description="Tóm tắt các điểm chính và các thực thể quan trọng từ lịch sử hội thoại để làm đầu vào cho các truy vấn phức tạp."
    )
    def summarize_context(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Tóm tắt một đoạn hội thoại.

        Args:
            conversation_history (List[Dict[str, Any]]): Lịch sử hội thoại cần tóm tắt.

        Returns:
            str: Một chuỗi văn bản tóm tắt các ý chính.
        """
        if not conversation_history:
            return "Không có lịch sử hội thoại."

        # Có thể dùng một LLM nhỏ để tóm tắt, hoặc đơn giản là nối chuỗi
        summary = "Tóm tắt hội thoại: "
        for turn in conversation_history:
            summary += f"{turn['role'].capitalize()}: {turn['content']}. "
        
        logger.info(f"Đã tóm tắt hội thoại thành: '{summary[:100]}...'")
        return summary

