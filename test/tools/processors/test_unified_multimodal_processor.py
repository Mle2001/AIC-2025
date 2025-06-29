# /test/tools/processors/test_unified_multimodal_processor.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from tools.processors.unified_multimodal_processor import (
    UnifiedMultiModalTool,
    VideoLanguageProcessor,
    AdvancedSpeechProcessor,
    AdvancedOCRProcessor
)
from tools.utils.scene_detection import SceneDetectionTool
from PIL import Image

@pytest.fixture
def mock_processors():
    with patch('tools.processors.unified_multimodal_processor.VideoLanguageProcessor') as mock_vlp_cls, \
         patch('tools.processors.unified_multimodal_processor.AdvancedSpeechProcessor') as mock_asp_cls, \
         patch('tools.processors.unified_multimodal_processor.AdvancedOCRProcessor') as mock_ocr_cls, \
         patch('tools.processors.unified_multimodal_processor.SceneDetectionTool') as mock_sdt_cls:

        mock_vlp = MagicMock(spec=VideoLanguageProcessor)
        mock_vlp.analyze_video_content = AsyncMock(return_value={"visual_narrative": "A cat is sleeping."})
        
        mock_asp = MagicMock(spec=AdvancedSpeechProcessor)
        mock_asp.transcribe_audio = AsyncMock(return_value={"text": "purring sound"})
        
        mock_ocr = MagicMock(spec=AdvancedOCRProcessor)
        mock_ocr.extract_text_from_frame = AsyncMock(return_value={"full_text": "CAT NAP"})
        
        mock_sdt = MagicMock(spec=SceneDetectionTool)
        mock_sdt.detect_scenes.return_value = [{"start_time": 0, "end_time": 10}]

        mock_vlp_cls.return_value = mock_vlp
        mock_asp_cls.return_value = mock_asp
        mock_ocr_cls.return_value = mock_ocr
        mock_sdt_cls.return_value = mock_sdt
        
        yield {"vlp": mock_vlp, "asp": mock_asp, "ocr": mock_ocr, "sdt": mock_sdt}

@pytest.mark.asyncio
# ✅ SỬA LỖI: Patch _extract_audio bằng MagicMock đồng bộ thông thường
@patch('tools.processors.unified_multimodal_processor.UnifiedMultiModalTool._extract_audio', new_callable=MagicMock)
@patch('tools.processors.unified_multimodal_processor.cv2.VideoCapture')
async def test_process_video_comprehensively_success(mock_videocapture, mock_extract_audio, mock_processors):
    tool = UnifiedMultiModalTool()
    video_path = "/fake/video.mp4"
    
    mock_extract_audio.return_value = "/fake/audio.mp3"
    mock_cap_inst = mock_videocapture.return_value
    mock_cap_inst.isOpened.return_value = True
    
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap_inst.read.return_value = (True, fake_frame)

    result = await tool.process_video_comprehensively(video_path)

    mock_processors["vlp"].analyze_video_content.assert_called_once_with(video_path)
    # Giờ đây transcribe_audio sẽ được gọi đúng với chuỗi đường dẫn
    mock_processors["asp"].transcribe_audio.assert_called_once_with("/fake/audio.mp3")
    mock_processors["ocr"].extract_text_from_frame.assert_called_once()
    assert result["summary"] == "Video-LLaVA: A cat is sleeping."
