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
        Khởi tạo tool. Có thể khởi tạo một LLM nhỏ hơn, chuyên dụng.
        """
        try:
            # Sử dụng một model nhỏ, nhanh và rẻ tiền cho tác vụ này
            self.question_generator_llm = OpenAIChat(id="gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo LLM cho việc tạo câu hỏi: {e}. Sẽ sử dụng logic dự phòng.")
            self.question_generator_llm = None
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
            video_references (List[Dict[str, Any]]): Danh sách các đoạn video liên quan.

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
    async def generate_follow_up_questions(self, response_context: Dict[str, Any]) -> List[str]:
        """
        Dựa vào nội dung câu trả lời và các video được tham chiếu để đưa ra các câu hỏi tiếp theo tiềm năng.

        Args:
            response_context (Dict[str, Any]): Ngữ cảnh của câu trả lời vừa được tạo.

        Returns:
            List[str]: Một danh sách các câu hỏi gợi ý.
        """
        logger.info("Đang tạo các câu hỏi gợi ý...")

        if not self.question_generator_llm:
            logger.warning("LLM không có sẵn, sử dụng logic dự phòng để tạo câu hỏi.")
            return ["Video nào liên quan nhất?", "Hãy cho tôi xem một cảnh khác."]

        try:
            text = response_context.get("text_response", "")
            # Tạo một prompt hiệu quả cho LLM
            prompt = f"""
            Based on the following AI response, generate 3 concise and relevant follow-up questions a user might ask.
            The response is about finding and describing video content.
            AI Response: "{text}"
            
            Return ONLY a Python list of strings. Example: ["Question 1?", "Question 2?", "Question 3?"]
            """
            
            # Gọi LLM
            raw_response = await self.question_generator_llm.invoke(prompt)
            
            # Xử lý và parse kết quả từ LLM
            # eval() có thể không an toàn, trong thực tế nên dùng ast.literal_eval
            import ast
            questions = ast.literal_eval(raw_response)
            
            if isinstance(questions, list):
                logger.info(f"Đã tạo {len(questions)} câu hỏi gợi ý từ LLM.")
                return questions
            else:
                logger.warning("LLM không trả về danh sách hợp lệ, sử dụng logic dự phòng.")
                return ["Mô tả chi tiết hơn.", "Có video nào khác không?"]

        except Exception as e:
            logger.error(f"Lỗi khi gọi LLM để tạo câu hỏi: {e}", exc_info=True)
            return ["Có lựa chọn nào khác không?"]
