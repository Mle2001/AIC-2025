# /test/tools/processors/test_unified_multimodal_processor.py
"""
Bộ kiểm thử cho UnifiedMultiModalTool mới.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Nhập các lớp mới cần kiểm thử và các lớp phụ thuộc
from tools.processors.unified_multimodal_processor import (
    UnifiedMultiModalTool,
    VideoLanguageProcessor,
    AdvancedSpeechProcessor,
    AdvancedOCRProcessor,
    SceneDetectionTool
)
from PIL import Image

# --- FIXTURES ---

@pytest.fixture
def mock_processors():
    """Fixture để mock tất cả các bộ xử lý con."""
    with patch('tools.processors.unified_multimodal_processor.VideoLanguageProcessor') as mock_vlp_cls, \
         patch('tools.processors.unified_multimodal_processor.AdvancedSpeechProcessor') as mock_asp_cls, \
         patch('tools.processors.unified_multimodal_processor.AdvancedOCRProcessor') as mock_ocr_cls, \
         patch('tools.processors.unified_multimodal_processor.SceneDetectionTool') as mock_sdt_cls:

        # Cấu hình các instance mock
        mock_vlp = MagicMock(spec=VideoLanguageProcessor)
        mock_vlp.analyze_video_content = AsyncMock(return_value={"visual_narrative": "A cat is sleeping."})
        
        mock_asp = MagicMock(spec=AdvancedSpeechProcessor)
        mock_asp.transcribe_audio = AsyncMock(return_value={"text": "purring sound"})
        
        mock_ocr = MagicMock(spec=AdvancedOCRProcessor)
        mock_ocr.extract_text_from_frame = AsyncMock(return_value={"full_text": "CAT NAP"})
        
        mock_sdt = MagicMock(spec=SceneDetectionTool)
        mock_sdt.detect_scenes.return_value = [{"start_time": 0, "end_time": 10}]

        # Gán các instance mock vào các class đã patch
        mock_vlp_cls.return_value = mock_vlp
        mock_asp_cls.return_value = mock_asp
        mock_ocr_cls.return_value = mock_ocr
        mock_sdt_cls.return_value = mock_sdt
        
        yield {
            "vlp": mock_vlp,
            "asp": mock_asp,
            "ocr": mock_ocr,
            "sdt": mock_sdt
        }

# --- TESTS ---

@pytest.mark.asyncio
async def test_process_video_comprehensively_success(mock_processors):
    """
    Kiểm thử tool chính 'process_video_comprehensively' trong trường hợp thành công.
    Mục tiêu: Đảm bảo nó gọi đúng các bộ xử lý con và hợp nhất kết quả.
    """
    # 1. ARRANGE
    tool = UnifiedMultiModalTool()
    video_path = "/fake/video.mp4"

    # 2. ACT
    result = await tool.process_video_comprehensively(video_path)

    # 3. ASSERT
    # Kiểm tra xem các bộ xử lý con có được gọi không
    mock_processors["vlp"].analyze_video_content.assert_called_once_with(video_path)
    mock_processors["asp"].transcribe_audio.assert_called_once()
    mock_processors["ocr"].extract_text_from_frame.assert_called_once()
    mock_processors["sdt"].detect_scenes.assert_called_once_with(video_path)

    # Kiểm tra cấu trúc của kết quả hợp nhất
    assert "summary" in result
    assert "full_transcript" in result
    assert "detected_text" in result
    assert "temporal_scenes" in result
    assert result["summary"] == "Video-LLaVA: A cat is sleeping."
    assert result["full_transcript"] == "purring sound"
    assert result["detected_text"] == "CAT NAP"
    assert len(result["temporal_scenes"]) == 1

