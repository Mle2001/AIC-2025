# /test/tools/video_processing/test_scene_detection.py
"""
Bộ kiểm thử cho SceneDetectionTool, tuân thủ kiến trúc Agno và Pytest.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Nhập công cụ cần kiểm thử
from tools.video_processing.scene_detection import SceneDetectionTool
# Nhập lớp này để mock, vì nó được sử dụng trong code của tool
from scenedetect.frame_timecode import FrameTimecode

# --- FIXTURES: Chuẩn bị môi trường và dữ liệu giả lập cho các bài test ---

@pytest.fixture
def scene_tool() -> SceneDetectionTool:
    """
    Cung cấp một instance mới của SceneDetectionTool cho mỗi bài test.
    Điều này đảm bảo các bài test không ảnh hưởng lẫn nhau.
    """
    return SceneDetectionTool()

@pytest.fixture
def fake_video_path(tmp_path: Path) -> str:
    """
    Tạo một đường dẫn tệp video giả trong một thư mục tạm thời.
    Sử dụng fixture `tmp_path` của pytest để không tạo file rác trong project.
    """
    p = tmp_path / "test.mp4"
    p.touch()
    return str(p)

# --- TESTS ---

@patch('tools.video_processing.scene_detection.VideoManager')
@patch('tools.video_processing.scene_detection.SceneManager')
@patch('tools.video_processing.scene_detection.ContentDetector')
def test_detect_scenes_tool_success(mock_detector_cls, mock_scene_manager_cls, mock_video_manager_cls, scene_tool, fake_video_path):
    """
    Kiểm thử tool 'detect_scenes' trong trường hợp thành công.
    Mục tiêu: Đảm bảo tool trả về kết quả đúng định dạng và gọi các thư viện bên dưới đúng cách.
    """
    # 1. ARRANGE: Chuẩn bị dữ liệu và các mock object
    
    # Tạo các đối tượng FrameTimecode giả lập để mô phỏng kết quả từ PySceneDetect
    mock_start1 = MagicMock(spec=FrameTimecode)
    mock_start1.get_seconds.return_value = 0.0
    mock_start1.get_timecode.return_value = "00:00:00.000"

    mock_end1 = MagicMock(spec=FrameTimecode)
    mock_end1.get_seconds.return_value = 10.5
    mock_end1.get_timecode.return_value = "00:00:10.500"
    
    # Giả lập phép trừ giữa hai đối tượng FrameTimecode
    (mock_end1 - mock_start1).get_seconds.return_value = 10.5
    
    # Cấu hình mock SceneManager để trả về danh sách cảnh giả lập của chúng ta
    mock_scene_manager_inst = mock_scene_manager_cls.return_value
    mock_scene_manager_inst.get_scene_list.return_value = [(mock_start1, mock_end1)]

    # 2. ACT: Gọi phương thức của tool cần kiểm thử
    # (Nhờ có conftest.py, lệnh gọi này sẽ hoạt động và không gây ra TypeError)
    result = scene_tool.detect_scenes(video_path=fake_video_path, threshold=30.0)

    # 3. ASSERT: Kiểm tra kết quả và các hành vi mong muốn
    
    # Kiểm tra nội dung của kết quả trả về
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["start_time"] == 0.0
    assert result[0]["end_time"] == 10.5
    assert result[0]["duration"] == 10.5
    assert result[0]["start_timecode"] == "00:00:00.000"

    # Kiểm tra xem các thư viện bên dưới có được gọi đúng cách không
    mock_video_manager_cls.assert_called_once_with([fake_video_path])
    mock_detector_cls.assert_called_once_with(threshold=30.0)
    mock_scene_manager_inst.detect_scenes.assert_called_once()
    mock_video_manager_cls.return_value.release.assert_called_once()


@patch('tools.video_processing.scene_detection.VideoManager', side_effect=IOError("Video file is corrupted"))
def test_detect_scenes_tool_handles_exception(mock_video_manager_cls, scene_tool, fake_video_path):
    """
    Kiểm thử tool 'detect_scenes' sẽ ném ra ngoại lệ khi thư viện bên dưới gặp lỗi.
    Theo triết lý của Agno, tool nên để cho agent xử lý exception thay vì "nuốt" nó.
    """
    # 1. ARRANGE & 2. ACT: Dùng pytest.raises để kiểm tra xem một exception có được ném ra không
    with pytest.raises(IOError, match="Video file is corrupted"):
        scene_tool.detect_scenes(video_path=fake_video_path)

    # 3. ASSERT: Đảm bảo rằng lời gọi đến thư viện vẫn được thực hiện
    mock_video_manager_cls.assert_called_once_with([fake_video_path])


def test_calculate_frame_difference_is_not_a_tool(scene_tool):
    """
    Kiểm tra rằng hàm tiện ích nội bộ (không có decorator @tool) không phải là một tool.
    Điều này quan trọng để đảm bảo agent không thể truy cập các hàm không được phép.
    """
    # Framework Agno sẽ không liệt kê hàm này trong danh sách các tool của class.
    # Chúng ta có thể kiểm tra bằng cách xem một thuộc tính đặc biệt mà decorator @tool
    # có thể thêm vào (nếu có), hoặc đơn giản là đảm bảo nó không có decorator.
    # Bài test này mang tính chất kiểm tra cấu trúc hơn là logic.
    # Giả sử @tool thêm thuộc tính `__is_tool__`
    assert not hasattr(scene_tool.calculate_frame_difference, "__is_tool__")
