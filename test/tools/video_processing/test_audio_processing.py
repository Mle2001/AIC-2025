# /test/tools/video_processing/test_audio_processing.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.video_processing.audio_processing import AudioProcessingTool

@pytest.fixture
def fake_video_path(tmp_path: Path) -> str:
    p = tmp_path / "test_video.mp4"
    p.touch()
    return str(p)

@pytest.fixture
def fake_audio_path(tmp_path: Path) -> str:
    p = tmp_path / "test_audio.mp3"
    p.touch()
    return str(p)

@pytest.fixture
def mock_dependencies():
    """Fixture để mock đồng thời cả whisper và moviepy."""
    with patch('tools.video_processing.audio_processing.whisper') as mock_whisper, \
         patch('tools.video_processing.audio_processing.VideoFileClip') as mock_moviepy:
        
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        mock_clip_instance = MagicMock()
        mock_audio_clip = MagicMock()
        mock_clip_instance.audio = mock_audio_clip
        mock_moviepy.return_value.__enter__.return_value = mock_clip_instance
        
        yield {"whisper": mock_whisper, "moviepy": mock_moviepy}

def test_transcribe_audio_tool_success(mock_dependencies, fake_video_path, fake_audio_path):
    """Kiểm thử tool 'transcribe_audio' trong trường hợp thành công."""
    tool = AudioProcessingTool()
    expected_result = {"text": "hello world", "segments": []}
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.return_value = expected_result
    
    result = tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
    
    assert result == expected_result
    mock_dependencies["moviepy"].assert_called_once_with(fake_video_path)
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio.write_audiofile.assert_called_once()
    mock_model.transcribe.assert_called_once_with(fake_audio_path, fp16=False)

def test_transcribe_audio_no_audio_stream(mock_dependencies, fake_video_path, fake_audio_path):
    """Kiểm thử trường hợp video không có luồng âm thanh."""
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio = None
    tool = AudioProcessingTool()
    
    result = tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
    
    assert result is None
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.assert_not_called()

def test_transcribe_audio_extraction_fails(mock_dependencies, fake_video_path, fake_audio_path):
    """Kiểm thử trường hợp trích xuất âm thanh thất bại."""
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio.write_audiofile.side_effect = Exception("Disk full")
    tool = AudioProcessingTool()
    
    with pytest.raises(Exception, match="Disk full"):
        tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
        
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.assert_not_called()

def test_lazy_loading_of_whisper_model(mock_dependencies):
    """Kiểm tra rằng mô hình Whisper chỉ được tải khi cần thiết."""
    mock_load_model = mock_dependencies["whisper"].load_model
    tool = AudioProcessingTool()
    
    mock_load_model.assert_not_called()
    
    # Truy cập thuộc tính lần đầu tiên
    model1 = tool.whisper_model
    mock_load_model.assert_called_once_with("base")
    
    # Truy cập lần thứ hai, không gọi lại load_model
    model2 = tool.whisper_model
    mock_load_model.assert_called_once()
    assert model1 is model2
