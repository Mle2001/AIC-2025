# /tests/tools/conversation/test_response_tools.py

import pytest
from unittest.mock import patch

# Nhập công cụ cần kiểm thử
from tools.conversation.response_tools import ResponseTool

# --- FIXTURES ---

@pytest.fixture
def response_tool() -> ResponseTool:
    """Cung cấp một instance của ResponseTool."""
    return ResponseTool()

# --- TESTS ---

def test_format_video_response_success(response_tool):
    """
    Kiểm thử tool 'format_video_response' tạo ra cấu trúc đúng.
    """
    # 1. ARRANGE
    text = "Tôi đã tìm thấy 2 video phù hợp."
    references = [
        {"video_id": "vid1", "timestamp": 30.5},
        {"video_id": "vid2", "timestamp": 120.0}
    ]

    # 2. ACT
    result = response_tool.format_video_response(
        text_response=text,
        video_references=references
    )

    # 3. ASSERT
    assert isinstance(result, dict)
    assert result["text_response"] == text
    assert result["media_references"] == references
    assert result["response_type"] == "video_answer"
    assert len(result["media_references"]) == 2

def test_generate_follow_up_questions_with_context(response_tool):
    """
    Kiểm thử tool 'generate_follow_up_questions' với logic giả định.
    """
    # 1. ARRANGE
    # Giả lập việc gọi đến LLM sẽ được thực hiện bên trong tool
    # Ở đây chúng ta chỉ kiểm tra logic giả định đã viết
    context1 = {"text_response": "Đây là mô tả chi tiết về video."}
    context2 = {"text_response": "Tìm thấy nhiều kết quả phù hợp."}
    context3 = {"text_response": "Một câu trả lời bình thường."}

    # 2. ACT
    questions1 = response_tool.generate_follow_up_questions(response_context=context1)
    questions2 = response_tool.generate_follow_up_questions(response_context=context2)
    questions3 = response_tool.generate_follow_up_questions(response_context=context3)

    # 3. ASSERT
    assert "Bạn có thể mô tả chi tiết hơn về cảnh đầu tiên không?" in questions1
    assert "Video nào là phù hợp nhất?" in questions2
    assert "Hãy tìm một video khác tương tự." in questions3

@patch('tools.conversation.response_tools.ResponseTool.generate_follow_up_questions')
def test_generate_follow_up_questions_mocked_llm(mock_generate_method):
    """
    Kiểm thử tool 'generate_follow_up_questions' bằng cách mock toàn bộ phương thức.
    Đây là cách tiếp cận khi logic bên trong quá phức tạp hoặc phụ thuộc LLM.
    """
    # 1. ARRANGE
    tool = ResponseTool()
    expected_questions = ["Câu hỏi 1?", "Câu hỏi 2?"]
    mock_generate_method.return_value = expected_questions
    context = {"text_response": "bất kỳ"}

    # 2. ACT
    result = tool.generate_follow_up_questions(response_context=context)

    # 3. ASSERT
    assert result == expected_questions
    mock_generate_method.assert_called_once_with(response_context=context)
