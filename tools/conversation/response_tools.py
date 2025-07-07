# /tools/conversation/response_tools.py
"""
Công cụ để định dạng và cấu trúc câu trả lời cuối cùng cho người dùng.
"""
import logging
from typing import Dict, List, Any

from agno.tools import tool

logger = logging.getLogger(__name__)

class ResponseTool:
    """
    Công cụ giúp ResponseSynthesisAgent tạo ra một câu trả lời hoàn chỉnh,
    kết hợp văn bản và các tham chiếu đa phương tiện.
    """

    def __init__(self):
        logger.info("ResponseTool đã được khởi tạo.")

    @tool(
        name="format_final_response_with_media",
        description="Định dạng câu trả lời cuối cùng, đính kèm các tham chiếu đến các cảnh video hoặc hình ảnh liên quan."
    )
    def format_final_response(
        self,
        text_answer: str,
        media_references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tạo một đối tượng phản hồi có cấu trúc.

        Args:
            text_answer (str): Nội dung văn bản chính của câu trả lời.
            media_references (List[Dict[str, Any]]): Danh sách các cảnh video liên quan,
                mỗi cảnh là một dict chứa 'scene_id', 'video_id', 'start_seconds', 'end_seconds'.

        Returns:
            Dict[str, Any]: Một đối tượng phản hồi hoàn chỉnh sẵn sàng để gửi cho người dùng.
        """
        logger.info("Đang định dạng câu trả lời cuối cùng với các tham chiếu media.")
        
        final_response = {
            "response_text": text_answer,
            "media_references": media_references,
            "suggested_questions": self._generate_follow_up_questions(text_answer),
            "response_type": "video_search_result" if media_references else "text_answer"
        }
        return final_response

    def _generate_follow_up_questions(self, context_text: str) -> List[str]:
        """
        Tạo ra các câu hỏi gợi ý dựa trên nội dung câu trả lời.
        (Đây là một phiên bản đơn giản, có thể được thay thế bằng một lệnh gọi LLM)
        """
        suggestions = [
            "Hãy giải thích chi tiết hơn về cảnh đầu tiên.",
            "Video này còn có những nội dung gì khác?"
        ]
        if "nấu ăn" in context_text:
            suggestions.append("Công thức cho món này là gì?")
        if "âm nhạc" in context_text:
            suggestions.append("Tên bài hát trong video là gì?")
            
        return suggestions[:3] # Trả về tối đa 3 gợi ý
