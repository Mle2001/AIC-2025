# /tools/conversation/context_tools.py
"""
Tool để quản lý và truy xuất ngữ cảnh hội thoại, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any
# ✅ SỬA LỖI: Sử dụng thư viện 'regex' thay cho 're' để hỗ trợ Unicode
import regex

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Giả định có một lớp CacheDB để tương tác với Redis
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
        """
        self.cache = cache_db
        logger.info("ContextTool đã được khởi tạo với CacheDB.")

    @tool(
        name="get_conversation_history",
        description="Lấy lịch sử của một cuộc hội thoại từ cache.",
        cache_results=False
    )
    async def get_conversation_history(self, session_id: str, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Truy xuất N lượt hội thoại cuối cùng cho một session ID nhất định.
        """
        logger.info(f"Đang lấy {num_turns} lượt hội thoại cuối cho session: {session_id}")
        try:
            history = await self.cache.get_chat_history(session_id, limit=num_turns)
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
        """
        logger.info(f"Đang trích xuất thực thể từ {len(conversation_history)} lượt hội thoại.")
        
        entities = []
        # ✅ SỬA LỖI: Regex này tìm các cụm từ có ít nhất một từ viết hoa,
        # và ưu tiên các cụm có nhiều từ. Nó cũng xử lý tiếng Việt có dấu.
        proper_noun_regex = r'\b\p{Lu}\p{L}*(?:\s+\p{Lu}\p{L}*)*\b'
        
        for turn in conversation_history:
            message = turn.get("message", "")
            if turn.get("role") == "user":
                # Tìm các cụm danh từ riêng
                found_proper_nouns = regex.findall(proper_noun_regex, message)
                for phrase in found_proper_nouns:
                    # Lọc bỏ các từ đơn đứng đầu câu không phải là danh từ riêng thực sự
                    if ' ' in phrase or phrase not in ["Hãy", "Tôi", "Thời"]:
                        if not any(e['name'].lower() == phrase.lower() for e in entities):
                            entities.append({"name": phrase, "type": "PROPER_NOUN"})
                
                # Tìm các cụm từ trong dấu ngoặc kép
                found_quoted = regex.findall(r'["\'](.*?)["\']', message)
                for phrase in found_quoted:
                     if not any(e['name'].lower() == phrase.lower() for e in entities):
                        entities.append({"name": phrase, "type": "QUOTED_PHRASE"})

        logger.info(f"Đã trích xuất được {len(entities)} thực thể từ ngữ cảnh.")
        return entities
