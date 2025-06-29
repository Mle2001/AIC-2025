# /tools/conversation/context_tools.py
"""
Tool để quản lý và truy xuất ngữ cảnh hội thoại, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Giả định có một lớp CacheDB để tương tác với Redis
# Lớp này sẽ được mock trong các bài test.
from database.connections.cache_db import CacheDB

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextTool:
    """
    Một class chứa các công cụ để giúp agent hiểu được ngữ cảnh của cuộc trò chuyện.
    """
    def __init__(self, cache_db: CacheDB):
        """
        Khởi tạo tool với một kết nối đến cache (Redis).
        Sử dụng Dependency Injection để dễ dàng kiểm thử.

        Args:
            cache_db (CacheDB): Instance của lớp quản lý cache.
        """
        self.cache = cache_db
        logger.info("ContextTool đã được khởi tạo với CacheDB.")

    @tool(
        name="get_conversation_history",
        description="Lấy lịch sử của một cuộc hội thoại từ cache.",
        cache_results=False # Không cache vì lịch sử luôn thay đổi
    )
    async def get_conversation_history(self, session_id: str, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Truy xuất N lượt hội thoại cuối cùng cho một session ID nhất định.

        Args:
            session_id (str): ID của phiên hội thoại.
            num_turns (int): Số lượng lượt hội thoại gần nhất cần lấy.

        Returns:
            List[Dict[str, Any]]: Một danh sách các lượt hội thoại, mỗi lượt là một dict.
        """
        logger.info(f"Đang lấy {num_turns} lượt hội thoại cuối cho session: {session_id}")
        try:
            history = await self.cache.get_chat_history(session_id, limit=num_turns)
            # Lịch sử từ Redis có thể được lưu ngược, đảo lại để đúng thứ tự thời gian
            history.reverse()
            return history
        except Exception as e:
            logger.error(f"Lỗi khi lấy lịch sử hội thoại cho session '{session_id}': {e}", exc_info=True)
            raise e

    @tool(
        name="extract_context_entities",
        description="Trích xuất các thực thể quan trọng từ lịch sử hội thoại.",
        cache_results=True,
        cache_ttl=300
    )
    def extract_context_entities(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phân tích lịch sử hội thoại để tìm các thực thể đã được đề cập trước đó.
        Ví dụ: nếu người dùng hỏi "Nó màu gì?" sau khi hỏi về "chiếc xe",
        tool này sẽ giúp xác định "nó" chính là "chiếc xe".

        Args:
            conversation_history (List[Dict]): Lịch sử các lượt hội thoại.

        Returns:
            List[Dict[str, Any]]: Một danh sách các thực thể được tìm thấy.
        """
        logger.info(f"Đang trích xuất thực thể từ {len(conversation_history)} lượt hội thoại.")
        
        # Trong thực tế, logic này sẽ phức tạp hơn nhiều, có thể gọi đến một LLM
        # để thực hiện phân giải đồng tham chiếu (coreference resolution).
        # Ví dụ đơn giản ở đây sẽ chỉ tìm các danh từ riêng.
        entities = []
        # Giả sử một logic đơn giản: tìm các từ viết hoa
        import re
        for turn in conversation_history:
            # Chỉ phân tích tin nhắn của người dùng
            if turn.get("role") == "user":
                message = turn.get("message", "")
                # Tìm các từ viết hoa (một cách giả định đơn giản cho entities)
                found_words = re.findall(r'\b[A-Z][a-z]*\b', message)
                for word in found_words:
                    if not any(e['name'] == word for e in entities):
                        entities.append({"name": word, "type": "PROPER_NOUN"})

        logger.info(f"Đã trích xuất được {len(entities)} thực thể từ ngữ cảnh.")
        return entities
