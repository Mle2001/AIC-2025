# /test/tools/video_processing/test_keyframe_extraction.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import cv2
import numpy as np

from tools.video_processing.keyframe_extraction import KeyframeExtractionTool

@pytest.fixture
def keyframe_tool() -> KeyframeExtractionTool:
    return KeyframeExtractionTool()

@pytest.fixture
def fake_video_path(tmp_path: Path) -> str:
    p = tmp_path / "test_video.mp4"
    p.touch()
    return str(p)

@pytest.fixture
def mock_opencv():
    """Fixture để mock các hàm của OpenCV."""
    with patch('tools.video_processing.keyframe_extraction.cv2.VideoCapture') as mock_vid_cap_cls, \
         patch('tools.video_processing.keyframe_extraction.cv2.imwrite') as mock_imwrite:
        
        mock_capture_inst = MagicMock()
        mock_capture_inst.isOpened.return_value = True
        mock_capture_inst.get.return_value = 30.0
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_capture_inst.read.return_value = (True, fake_frame)
        mock_vid_cap_cls.return_value = mock_capture_inst
        
        yield {"cap": mock_vid_cap_cls, "write": mock_imwrite}

def test_extract_keyframes_tool_success(keyframe_tool, fake_video_path, tmp_path, mock_opencv):
    """Kiểm thử tool 'extract_keyframes' trong trường hợp thành công."""
    scenes_data = [
        {"start_time": 10.0, "end_time": 20.0},
        {"start_time": 30.0, "end_time": 40.0},
    ]
    output_dir = str(tmp_path / "output")

    result = keyframe_tool.extract_keyframes(
        video_path=fake_video_path,
        scenes=scenes_data,
        output_dir=output_dir
    )

    assert len(result) == 2
    assert result[0]["timestamp"] == 15.0
    assert result[1]["timestamp"] == 35.0
    assert mock_opencv["write"].call_count == 2
    
    mock_capture_inst = mock_opencv["cap"].return_value
    calls = mock_capture_inst.set.call_args_list
    assert call(cv2.CAP_PROP_POS_FRAMES, 450) in calls
    assert call(cv2.CAP_PROP_POS_FRAMES, 1050) in calls

def test_extract_keyframes_video_open_fails(keyframe_tool, fake_video_path, mock_opencv):
    """Kiểm thử tool ném ra IOError khi không thể mở video."""
    mock_opencv["cap"].return_value.isOpened.return_value = False
    with pytest.raises(IOError, match="Could not open video file"):
        keyframe_tool.extract_keyframes(video_path=fake_video_path, scenes=[])

def test_extract_keyframes_with_empty_scenes(keyframe_tool, fake_video_path, mock_opencv):
    """Kiểm thử tool trả về danh sách rỗng khi không có cảnh nào được cung cấp."""
    result = keyframe_tool.extract_keyframes(video_path=fake_video_path, scenes=[])
    assert result == []
    mock_opencv["write"].assert_not_called()

def test_extract_keyframes_read_frame_fails(keyframe_tool, fake_video_path, mock_opencv):
    """Kiểm thử tool bỏ qua cảnh nếu không thể đọc được khung hình."""
    mock_opencv["cap"].return_value.read.return_value = (False, None)
    scenes_data = [{"start_time": 10.0, "end_time": 20.0}]
    result = keyframe_tool.extract_keyframes(video_path=fake_video_path, scenes=scenes_data)
    assert result == []
    mock_opencv["write"].assert_not_called()
