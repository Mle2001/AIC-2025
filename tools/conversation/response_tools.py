# /tools/conversation/response_tools.py
"""
Tool để định dạng và tạo các thành phần cho câu trả lời, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Giả định có thể gọi đến một LLM để tạo câu hỏi gợi ý
from agno.models.openai import OpenAIChat

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseTool:
    """
    Một class chứa các công cụ để định dạng câu trả lời cuối cùng
    và tạo ra các câu hỏi gợi ý.
    """
    def __init__(self):
        """
        Khởi tạo tool.
        """
        # Có thể khởi tạo một LLM nhỏ hơn, chuyên dụng cho việc tạo câu hỏi
        # self.question_generator_llm = OpenAIChat(id="gpt-3.5-turbo")
        logger.info("ResponseTool đã được khởi tạo.")

    @tool(
        name="format_video_response",
        description="Định dạng câu trả lời cuối cùng, đính kèm các tham chiếu video có liên quan.",
        cache_results=False # Không cache vì nội dung trả lời luôn động
    )
    def format_video_response(
        self,
        text_response: str,
        video_references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Kết hợp văn bản trả lời và các đoạn video tham chiếu thành một cấu trúc thống nhất.

        Args:
            text_response (str): Nội dung văn bản chính của câu trả lời.
            video_references (List[Dict[str, Any]]): Danh sách các đoạn video liên quan,
                                                     mỗi dict chứa thông tin như video_id, timestamp, v.v.

        Returns:
            Dict[str, Any]: Một dictionary chứa câu trả lời đã được định dạng.
        """
        logger.info(f"Đang định dạng câu trả lời với {len(video_references)} tham chiếu video.")
        
        formatted_response = {
            "text_response": text_response,
            "media_references": video_references,
            "response_type": "video_answer"
        }
        return formatted_response

    @tool(
        name="generate_follow_up_questions",
        description="Tạo ra các câu hỏi gợi ý dựa trên ngữ cảnh của câu trả lời hiện tại.",
        cache_results=True,
        cache_ttl=60 # Cache các câu hỏi gợi ý trong 1 phút
    )
    def generate_follow_up_questions(self, response_context: Dict[str, Any]) -> List[str]:
        """
        Dựa vào nội dung câu trả lời và các video được tham chiếu để đưa ra các câu hỏi tiếp theo tiềm năng.

        Args:
            response_context (Dict[str, Any]): Ngữ cảnh của câu trả lời vừa được tạo,
                                                bao gồm cả text và media.

        Returns:
            List[str]: Một danh sách các câu hỏi gợi ý.
        """
        logger.info("Đang tạo các câu hỏi gợi ý...")

        # Trong thực tế, logic này sẽ gọi đến một LLM.
        # Ví dụ:
        # prompt = f"Dựa trên câu trả lời sau: '{response_context.get('text_response')}', 
        #           và các video về '{response_context.get('main_topic')}', 
        #           hãy tạo 3 câu hỏi tiếp theo mà người dùng có thể hỏi.
        #           Chỉ trả về danh sách các câu hỏi."
        # questions = self.question_generator_llm.invoke(prompt)
        
        # Logic giả định đơn giản cho mục đích minh họa
        text = response_context.get("text_response", "")
        questions = []
        if "mô tả" in text.lower():
            questions.append("Bạn có thể mô tả chi tiết hơn về cảnh đầu tiên không?")
        if "nhiều kết quả" in text.lower():
            questions.append("Video nào là phù hợp nhất?")
        if not questions:
            questions.append("Hãy tìm một video khác tương tự.")
            
        logger.info(f"Đã tạo {len(questions)} câu hỏi gợi ý.")
        return questions
