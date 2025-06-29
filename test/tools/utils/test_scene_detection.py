# /test/utils/test_scene_detection.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.utils.scene_detection import SceneDetectionTool
from scenedetect.frame_timecode import FrameTimecode

@pytest.fixture
def scene_tool() -> SceneDetectionTool:
    return SceneDetectionTool()

@pytest.fixture
def fake_video_path(tmp_path: Path) -> str:
    p = tmp_path / "test.mp4"
    p.touch()
    return str(p)

@patch('tools.utils.scene_detection.VideoManager')
@patch('tools.utils.scene_detection.SceneManager')
@patch('tools.utils.scene_detection.ContentDetector')
def test_detect_scenes_tool_success(mock_detector_cls, mock_scene_manager_cls, mock_video_manager_cls, scene_tool, fake_video_path):
    mock_start1 = MagicMock(spec=FrameTimecode); mock_start1.get_seconds.return_value = 0.0; mock_start1.get_timecode.return_value = "00:00:00.000"
    mock_end1 = MagicMock(spec=FrameTimecode); mock_end1.get_seconds.return_value = 10.5; mock_end1.get_timecode.return_value = "00:00:10.500"
    (mock_end1 - mock_start1).get_seconds.return_value = 10.5
    mock_scene_manager_inst = mock_scene_manager_cls.return_value
    mock_scene_manager_inst.get_scene_list.return_value = [(mock_start1, mock_end1)]
    
    result = scene_tool.detect_scenes(video_path=fake_video_path, threshold=30.0)
    
    assert len(result) == 1
    assert result[0]["start_time"] == 0.0
    mock_video_manager_cls.assert_called_once_with([fake_video_path])
    mock_detector_cls.assert_called_once_with(threshold=30.0)
