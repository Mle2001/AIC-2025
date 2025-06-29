# /tools/utils/scene_detection.py
"""
Công cụ tiện ích cấp thấp để phát hiện cảnh trong video.
"""
import logging
from typing import List, Dict, Any
from agno.tools import tool
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SceneDetectionTool:
    @tool(
        name="detect_scenes",
        description="Phát hiện các ranh giới cảnh trong một tệp video.",
        cache_results=True,
        cache_ttl=7200
    )
    def detect_scenes(self, video_path: str, threshold: float = 27.0) -> List[Dict[str, Any]]:
        logger.info(f"Bắt đầu phát hiện cảnh cho video: {video_path} với threshold={threshold}")
        video_manager = None
        try:
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list_timecodes = scene_manager.get_scene_list()
            if not scene_list_timecodes:
                logger.warning(f"Không tìm thấy cảnh nào trong video: {video_path}")
                return []
            
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
            raise e
        finally:
            if video_manager:
                video_manager.release()
