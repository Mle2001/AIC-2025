# /tests/tools/video_processing/test_audio_processing.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Nhập công cụ cần kiểm thử
from tools.video_processing.audio_processing import AudioProcessingTool

# --- FIXTURES ---

@pytest.fixture
def fake_video_path(tmp_path: Path) -> str:
    """Tạo một đường dẫn tệp video giả."""
    p = tmp_path / "test.mp4"
    p.touch()
    return str(p)

@pytest.fixture
def fake_audio_path(tmp_path: Path) -> str:
    """Tạo một đường dẫn tệp âm thanh giả."""
    p = tmp_path / "audio.mp3"
    p.touch()
    return str(p)

@pytest.fixture
def mock_dependencies():
    """Fixture để mock đồng thời cả whisper và moviepy."""
    with patch('tools.video_processing.audio_processing.whisper') as mock_whisper, \
         patch('tools.video_processing.audio_processing.VideoFileClip') as mock_moviepy:
        
        # Cấu hình whisper mock
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        
        # Cấu hình moviepy mock
        mock_clip_instance = MagicMock()
        mock_audio_clip = MagicMock()
        mock_clip_instance.audio = mock_audio_clip
        mock_moviepy.return_value.__enter__.return_value = mock_clip_instance
        
        yield {"whisper": mock_whisper, "moviepy": mock_moviepy}

# --- TESTS ---

def test_transcribe_audio_tool_success(mock_dependencies, fake_video_path, fake_audio_path):
    """
    Kiểm thử tool 'transcribe_audio' trong trường hợp thành công.
    """
    # 1. ARRANGE
    tool = AudioProcessingTool()
    
    # Dữ liệu giả lập trả về từ Whisper
    expected_result = {"text": "hello world", "segments": []}
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.return_value = expected_result
    
    # 2. ACT
    result = tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
    
    # 3. ASSERT
    assert result == expected_result
    
    # Kiểm tra các lời gọi mock
    mock_dependencies["moviepy"].assert_called_once_with(fake_video_path)
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio.write_audiofile.assert_called_once()
    
    mock_model.transcribe.assert_called_once_with(fake_audio_path, fp16=False)

def test_transcribe_audio_no_audio_stream(mock_dependencies, fake_video_path, fake_audio_path):
    """
    Kiểm thử trường hợp video không có luồng âm thanh.
    """
    # 1. ARRANGE
    # Cấu hình lại moviepy mock để không có audio
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio = None
    
    tool = AudioProcessingTool()
    
    # 2. ACT
    result = tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
    
    # 3. ASSERT
    assert result is None
    # Đảm bảo hàm transcribe không được gọi
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.assert_not_called()

def test_transcribe_audio_extraction_fails(mock_dependencies, fake_video_path, fake_audio_path):
    """
    Kiểm thử trường hợp trích xuất âm thanh thất bại.
    """
    # 1. ARRANGE
    mock_clip_instance = mock_dependencies["moviepy"].return_value.__enter__.return_value
    mock_clip_instance.audio.write_audiofile.side_effect = Exception("Disk full")
    
    tool = AudioProcessingTool()
    
    # 2. ACT & 3. ASSERT
    with pytest.raises(Exception, match="Disk full"):
        tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)
        
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.assert_not_called()

def test_transcribe_audio_transcription_fails(mock_dependencies, fake_video_path, fake_audio_path):
    """
    Kiểm thử trường hợp chuyển đổi văn bản thất bại.
    """
    # 1. ARRANGE
    mock_model = mock_dependencies["whisper"].load_model.return_value
    mock_model.transcribe.side_effect = Exception("Whisper error")
    
    tool = AudioProcessingTool()

    # 2. ACT & 3. ASSERT
    with pytest.raises(Exception, match="Whisper error"):
        tool.transcribe_audio(video_path=fake_video_path, audio_output_path=fake_audio_path)

def test_lazy_loading_of_whisper_model(mock_dependencies):
    """
    Kiểm tra rằng mô hình Whisper chỉ được tải khi cần thiết (lazy loading).
    """
    # 1. ARRANGE
    mock_load_model = mock_dependencies["whisper"].load_model
    tool = AudioProcessingTool()
    
    # 2. ASSERT (lần 1)
    # Lúc này, chỉ khởi tạo tool, chưa gọi đến model, nên load_model chưa được gọi
    mock_load_model.assert_not_called()
    
    # 3. ACT
    # Truy cập thuộc tính whisper_model lần đầu tiên
    model1 = tool.whisper_model
    
    # 4. ASSERT (lần 2)
    # Bây giờ load_model phải được gọi đúng 1 lần
    mock_load_model.assert_called_once_with("base")
    
    # 5. ACT
    # Truy cập thuộc tính lần thứ hai
    model2 = tool.whisper_model
    
    # 6. ASSERT (lần 3)
    # Số lần gọi load_model vẫn phải là 1, vì kết quả đã được cache lại
    mock_load_model.assert_called_once()
    assert model1 is model2 # Đảm bảo trả về cùng một instance
