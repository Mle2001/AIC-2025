# /tools/video_processing/audio_processing.py
"""
Tool cho audio processing và speech-to-text sử dụng Whisper, tuân thủ kiến trúc Agno.
Công cụ này được duy trì để sử dụng song song hoặc làm phương án dự phòng.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện xử lý
from moviepy.editor import VideoFileClip
import whisper

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessingTool:
    """
    Một class chứa các công cụ để trích xuất và phân tích âm thanh từ video bằng Whisper.
    """
    def __init__(self, whisper_model_name: str = "base"):
        """
        Khởi tạo tool và tải mô hình Whisper.
        """
        self._whisper_model_name = whisper_model_name
        self._model = None

    @property
    def whisper_model(self):
        """
        Sử dụng property để tải mô hình một cách lười biếng (lazy loading).
        """
        if self._model is None:
            logger.info(f"Đang tải mô hình Whisper: '{self._whisper_model_name}'...")
            try:
                self._model = whisper.load_model(self._whisper_model_name)
                logger.info("Tải mô hình Whisper thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình Whisper '{self._whisper_model_name}': {e}", exc_info=True)
                raise RuntimeError(f"Không thể tải mô hình Whisper: {e}")
        return self._model

    @tool(
        name="transcribe_audio_legacy",
        description="Trích xuất âm thanh từ video và chuyển đổi thành văn bản bằng Whisper.",
        cache_results=True,
        cache_ttl=7200
    )
    def transcribe_audio(self, video_path: str, audio_output_path: str) -> Optional[Dict[str, Any]]:
        """
        Thực hiện hai bước: trích xuất âm thanh và sau đó chuyển đổi thành văn bản.

        Args:
            video_path (str): Đường dẫn đến tệp video nguồn.
            audio_output_path (str): Đường dẫn để lưu tệp âm thanh được trích xuất.

        Returns:
            Optional[Dict[str, Any]]: Kết quả từ Whisper, hoặc None nếu không có âm thanh.
        """
        logger.info(f"Bắt đầu xử lý âm thanh (legacy) cho video: {video_path}")
        
        audio_path = self._extract_audio(video_path, audio_output_path)
        
        if not audio_path:
            logger.warning(f"Video {video_path} không có âm thanh. Bỏ qua chuyển đổi.")
            return None

        logger.info(f"Bắt đầu chuyển đổi văn bản cho tệp âm thanh: {audio_path}")
        try:
            result = self.whisper_model.transcribe(audio_path, fp16=False)
            logger.info(f"Chuyển đổi văn bản thành công cho: {audio_path}")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi văn bản cho tệp âm thanh '{audio_path}': {e}", exc_info=True)
            raise e

    def _extract_audio(self, video_path: str, output_path: str) -> Optional[str]:
        """Hàm tiện ích nội bộ để trích xuất âm thanh."""
        video_file = Path(video_path)
        audio_file = Path(output_path)
        audio_file.parent.mkdir(parents=True, exist_ok=True)

        if not video_file.exists():
            raise FileNotFoundError(f"Tệp video không tồn tại: {video_path}")

        try:
            with VideoFileClip(str(video_file)) as video_clip:
                if video_clip.audio is None:
                    return None
                video_clip.audio.write_audiofile(str(audio_file), codec='mp3', logger=None)
            return str(audio_file)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất âm thanh từ '{video_file.name}': {e}", exc_info=True)
            raise e
