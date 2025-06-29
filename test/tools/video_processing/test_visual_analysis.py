# /test/tools/video_processing/test_visual_analysis.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from tools.video_processing.visual_analysis import VisualAnalysisTool

@pytest.fixture
def fake_image_path(tmp_path: Path) -> str:
    p = tmp_path / "test_image.jpg"
    p.touch()
    return str(p)

@pytest.fixture
def mock_dependencies():
    """Fixture để mock đồng thời cả BLIP và DETR."""
    with patch('tools.video_processing.visual_analysis.Image') as mock_image_cls, \
         patch('tools.video_processing.visual_analysis.torch') as mock_torch, \
         patch('tools.video_processing.visual_analysis.BlipProcessor') as mock_blip_proc_cls, \
         patch('tools.video_processing.visual_analysis.BlipForConditionalGeneration') as mock_blip_model_cls, \
         patch('tools.video_processing.visual_analysis.DetrImageProcessor') as mock_detr_proc_cls, \
         patch('tools.video_processing.visual_analysis.DetrForObjectDetection') as mock_detr_model_cls:

        # Cấu hình chung
        mock_torch.cuda.is_available.return_value = False
        mock_image_cls.open.return_value.convert.return_value = MagicMock()

        # Cấu hình mock cho BLIP (Captioning)
        mock_blip_proc_inst = MagicMock()
        mock_blip_model_inst = MagicMock()
        mock_blip_proc_cls.from_pretrained.return_value = mock_blip_proc_inst
        mock_blip_model_cls.from_pretrained.return_value.to.return_value = mock_blip_model_inst
        
        # Cấu hình mock cho DETR (Detection)
        mock_detr_proc_inst = MagicMock()
        mock_detr_model_inst = MagicMock()
        # Giả lập thuộc tính config.id2label
        mock_detr_model_inst.config.id2label = {1: 'cat', 2: 'dog'}
        mock_detr_proc_cls.from_pretrained.return_value = mock_detr_proc_inst
        mock_detr_model_cls.from_pretrained.return_value.to.return_value = mock_detr_model_inst
        
        yield {
            "blip_proc": mock_blip_proc_inst,
            "blip_model": mock_blip_model_inst,
            "detr_proc": mock_detr_proc_inst,
            "detr_model": mock_detr_model_inst,
        }

def test_detect_objects_success(mock_dependencies, fake_image_path):
    """Kiểm thử tool 'detect_objects' với logic DETR hoàn chỉnh."""
    tool = VisualAnalysisTool()
    
    # ARRANGE: Cấu hình dữ liệu trả về giả lập từ DETR
    mock_detr_proc = mock_dependencies["detr_proc"]
    mock_detr_model = mock_dependencies["detr_model"]
    
    # Dữ liệu trả về thô từ model
    raw_outputs = {
        "scores": torch.tensor([0.99, 0.95]),
        "labels": torch.tensor([1, 2]),
        "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
    }
    mock_detr_proc.post_process_object_detection.return_value = [raw_outputs]

    # ACT
    result = tool.detect_objects(image_path=fake_image_path)

    # ASSERT
    assert len(result) == 2
    assert result[0]["label"] == "cat"
    assert result[0]["confidence"] == 0.99
    assert result[0]["box"] == [10.0, 10.0, 50.0, 50.0]
    assert result[1]["label"] == "dog"
    mock_detr_model.assert_called_once()
    mock_detr_proc.post_process_object_detection.assert_called_once()


def test_analyze_image_calls_both_models(mock_dependencies, fake_image_path):
    """Kiểm thử tool 'analyze_image' gọi đến cả BLIP và DETR."""
    tool = VisualAnalysisTool()

    # ARRANGE: Cấu hình dữ liệu trả về giả lập
    mock_dependencies["blip_model"].generate.return_value = [torch.tensor([1])]
    mock_dependencies["blip_proc"].decode.return_value = "a test caption"
    mock_dependencies["detr_proc"].post_process_object_detection.return_value = [{"scores": [], "labels": [], "boxes": []}]

    # ACT
    result = tool.analyze_image(image_path=fake_image_path)

    # ASSERT
    assert result["description"] == "a test caption"
    assert result["objects"] == []
    
    # Kiểm tra xem cả hai model có được gọi không
    mock_dependencies["blip_model"].generate.assert_called_once()
    mock_dependencies["detr_model"].assert_called_once()
