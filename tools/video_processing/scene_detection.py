# /tools/video_processing/scene_detection.py
"""
Tool cho scene detection trong videos, tuân thủ kiến trúc Agno.
"""
import logging
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện xử lý video
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SceneDetectionTool:
    """
    Một class chứa các công cụ liên quan đến việc phát hiện cảnh trong video.
    Mỗi phương thức được trang trí bằng @tool sẽ có thể được agent sử dụng.
    """

    @tool(
        name="detect_scenes",
        description="Phát hiện các ranh giới cảnh trong một tệp video và trả về danh sách các cảnh.",
        cache_results=True,  # Kích hoạt caching cho tool này
        cache_ttl=7200       # Cache kết quả trong 2 giờ
    )
    def detect_scenes(self, video_path: str, threshold: float = 27.0) -> List[Dict[str, Any]]:
        """
        Sử dụng PySceneDetect để phân tích video và tìm các điểm chuyển cảnh.

        Args:
            video_path (str): Đường dẫn tuyệt đối đến tệp video cần phân tích.
            threshold (float): Ngưỡng nhạy để phát hiện thay đổi. Giá trị thấp hơn sẽ phát hiện nhiều cảnh hơn.

        Returns:
            List[Dict[str, Any]]: Một danh sách các dictionary, mỗi dict đại diện cho một cảnh,
                                  chứa 'start_time', 'end_time', và 'duration' tính bằng giây.
                                  Trả về danh sách rỗng nếu có lỗi.
        """
        logger.info(f"Bắt đầu phát hiện cảnh cho video: {video_path} với threshold={threshold}")
        video_manager = None
        try:
            # Khởi tạo VideoManager để đọc video
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()

            # Thêm detector để tìm kiếm sự thay đổi nội dung
            scene_manager.add_detector(ContentDetector(threshold=threshold))

            # Bắt đầu xử lý
            video_manager.set_downscale_factor()
            video_manager.start()

            # Thực hiện phát hiện cảnh
            scene_manager.detect_scenes(frame_source=video_manager)

            # Lấy danh sách các cảnh dưới dạng FrameTimecode
            scene_list_timecodes = scene_manager.get_scene_list()

            if not scene_list_timecodes:
                logger.warning(f"Không tìm thấy cảnh nào trong video: {video_path}")
                return []

            # Chuyển đổi kết quả sang định dạng dictionary với đơn vị là giây
            formatted_scenes = []
            for start_timecode, end_timecode in scene_list_timecodes:
                scene_data = {
                    "start_time": start_timecode.get_seconds(),
                    "end_time": end_timecode.get_seconds(),
                    "duration": (end_timecode - start_timecode).get_seconds(),
                    "start_timecode": start_timecode.get_timecode(),
                    "end_timecode": end_timecode.get_timecode(),
                }
                formatted_scenes.append(scene_data)

            logger.info(f"Phát hiện thành công {len(formatted_scenes)} cảnh.")
            return formatted_scenes

        except Exception as e:
            logger.error(f"Lỗi khi phát hiện cảnh cho video '{video_path}': {e}", exc_info=True)
            # Theo hướng dẫn của Agno, tool nên ném ra ngoại lệ để agent có thể xử lý
            raise e
        finally:
            if video_manager:
                video_manager.release()

    def calculate_frame_difference(self, frame1, frame2) -> float:
        """
        Hàm tiện ích nội bộ (không phải là một tool) để tính toán sự khác biệt giữa hai khung hình.
        Hàm này không được trang trí bằng @tool nên agent không thể gọi trực tiếp.

        Args:
            frame1, frame2: Dữ liệu khung hình dạng numpy array.

        Returns:
            float: Điểm số khác biệt.
        """
        # Implementation logic...
        # (Giữ lại logic từ trước nếu cần)
        pass
