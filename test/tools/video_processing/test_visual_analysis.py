# /tests/tools/video_processing/test_visual_analysis.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch  # <-- THÊM DÒNG NÀY

# Nhập công cụ cần kiểm thử
from tools.video_processing.visual_analysis import VisualAnalysisTool

# --- FIXTURES ---

@pytest.fixture
def fake_image_path(tmp_path: Path) -> str:
    """Tạo một đường dẫn tệp ảnh giả."""
    p = tmp_path / "test_image.jpg"
    p.touch()
    return str(p)

@pytest.fixture
def mock_dependencies():
    """Fixture để mock các thư viện AI bên ngoài (PIL, torch, transformers)."""
    with patch('tools.video_processing.visual_analysis.Image') as mock_image_cls, \
         patch('tools.video_processing.visual_analysis.torch') as mock_torch, \
         patch('tools.video_processing.visual_analysis.BlipProcessor') as mock_processor_cls, \
         patch('tools.video_processing.visual_analysis.BlipForConditionalGeneration') as mock_model_cls:

        # Cấu hình mock cho torch
        mock_torch.cuda.is_available.return_value = False # Giả lập chạy trên CPU

        # Cấu hình mock cho transformers
        mock_processor_inst = MagicMock()
        mock_model_inst = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor_inst
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model_inst

        # Cấu hình mock cho PIL
        mock_image_inst = MagicMock()
        mock_image_cls.open.return_value.convert.return_value = mock_image_inst
        
        yield {
            "Image": mock_image_cls,
            "torch": mock_torch,
            "BlipProcessor": mock_processor_cls,
            "BlipForConditionalGeneration": mock_model_cls,
            "processor_inst": mock_processor_inst,
            "model_inst": mock_model_inst
        }

# --- TESTS ---

def test_analyze_image_tool_success(mock_dependencies, fake_image_path):
    """
    Kiểm thử tool 'analyze_image' trong trường hợp thành công.
    """
    # 1. ARRANGE
    tool = VisualAnalysisTool()
    
    # Cấu hình dữ liệu trả về giả lập từ các mock
    mock_model_inst = mock_dependencies["model_inst"]
    mock_processor_inst = mock_dependencies["processor_inst"]
    expected_caption = "a dog playing in a park"
    
    # Dòng này sẽ không còn gây lỗi NameError
    mock_model_inst.generate.return_value = [torch.tensor([1, 2, 3])] 
    mock_processor_inst.decode.return_value = expected_caption

    # 2. ACT
    result = tool.analyze_image(image_path=fake_image_path)

    # 3. ASSERT
    assert isinstance(result, dict)
    assert result["description"] == expected_caption
    # Kiểm tra kết quả giả định từ tool detect_objects
    assert len(result["objects"]) == 2
    assert result["objects"][0]["label"] == "person"
    
    # Kiểm tra các lời gọi mock
    mock_dependencies["BlipProcessor"].from_pretrained.assert_called_once()
    mock_dependencies["BlipForConditionalGeneration"].from_pretrained.assert_called_once()
    mock_dependencies["Image"].open.assert_called_once_with(fake_image_path)
    mock_model_inst.generate.assert_called_once()


def test_analyze_image_file_not_found(mock_dependencies):
    """
    Kiểm thử tool ném ra FileNotFoundError khi ảnh không tồn tại.
    """
    # 1. ARRANGE
    tool = VisualAnalysisTool()
    
    # 2. ACT & 3. ASSERT
    with pytest.raises(FileNotFoundError):
        tool.analyze_image(image_path="/non/existent/path.jpg")

@patch('tools.video_processing.visual_analysis.BlipProcessor.from_pretrained', side_effect=RuntimeError("Hugging Face Hub is down"))
def test_tool_handles_model_loading_failure(mock_from_pretrained, fake_image_path):
    """
    Kiểm thử tool ném ra RuntimeError nếu không thể tải được mô hình.
    """
    # 1. ARRANGE
    tool = VisualAnalysisTool()

    # 2. ACT & 3. ASSERT
    with pytest.raises(RuntimeError, match="Không thể tải mô hình Visual Analysis"):
        tool.analyze_image(image_path=fake_image_path)

def test_detect_objects_tool_is_callable(fake_image_path):
    """
    Kiểm tra rằng tool 'detect_objects' có thể được gọi và trả về kết quả giả định.
    """
    # 1. ARRANGE
    tool = VisualAnalysisTool()
    
    # 2. ACT
    result = tool.detect_objects(image_path=fake_image_path)
    
    # 3. ASSERT
    assert isinstance(result, list)
    assert len(result) > 0
    assert "label" in result[0]
