# /test/tools/conversation/test_response_tools.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from tools.conversation.response_tools import ResponseTool

@pytest.fixture
def response_tool() -> ResponseTool:
    """Cung cấp một instance của ResponseTool."""
    return ResponseTool()

def test_format_video_response_success(response_tool):
    """Kiểm thử tool 'format_video_response' tạo ra cấu trúc đúng."""
    text = "Tôi đã tìm thấy 2 video phù hợp."
    references = [
        {"video_id": "vid1", "timestamp": 30.5},
        {"video_id": "vid2", "timestamp": 120.0}
    ]

    result = response_tool.format_video_response(
        text_response=text,
        video_references=references
    )

    assert isinstance(result, dict)
    assert result["text_response"] == text
    assert result["media_references"] == references
    assert result["response_type"] == "video_answer"

@pytest.mark.asyncio
@patch('tools.conversation.response_tools.OpenAIChat')
async def test_generate_follow_up_questions_with_mocked_llm(mock_llm_cls):
    """Kiểm thử tool 'generate_follow_up_questions' khi gọi LLM thành công."""
    # ARRANGE
    mock_llm_instance = MagicMock()
    # Giả lập LLM trả về một chuỗi dạng list
    mock_llm_instance.invoke = AsyncMock(return_value='["Câu hỏi 1 từ LLM?", "Câu hỏi 2?"]')
    mock_llm_cls.return_value = mock_llm_instance
    
    tool = ResponseTool()
    context = {"text_response": "Đây là video về chó và mèo."}

    # ACT
    result = await tool.generate_follow_up_questions(response_context=context)

    # ASSERT
    assert result == ["Câu hỏi 1 từ LLM?", "Câu hỏi 2?"]
    mock_llm_instance.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_generate_follow_up_questions_llm_fails(response_tool):
    """Kiểm thử tool sử dụng logic dự phòng khi LLM bị lỗi."""
    # ARRANGE
    # Giả lập LLM bị lỗi khi khởi tạo
    response_tool.question_generator_llm = None
    context = {"text_response": "bất kỳ"}

    # ACT
    result = await response_tool.generate_follow_up_questions(response_context=context)

    # ASSERT
    # Kiểm tra xem nó có trả về danh sách câu hỏi dự phòng không
    assert isinstance(result, list)
    assert len(result) == 2
    assert "Video nào liên quan nhất?" in result
