# /tools/utils/scene_detection.py
"""
Module tiện ích cho việc phát hiện và tách cảnh từ video.
Sử dụng PySceneDetect để thực hiện các tác vụ nặng.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# PySceneDetect, thư viện chuyên dụng cho phân cảnh
from scenedetect import detect, ContentDetector, SceneManager
from scenedetect.video_splitter import split_video_ffmpeg

# Cấu hình logging
logger = logging.getLogger(__name__)

class SceneDetectionTool:
    """
    Công cụ chuyên dụng để phát hiện và tách video thành các file cảnh riêng lẻ.
    Đây là bước đầu tiên trong pipeline tiền xử lý (Phase 1).
    """

    def split_video_into_scenes(self, video_path: str, output_dir: str, video_id: str) -> List[Dict[str, Any]]:
        """
        Phân tích một video, phát hiện các cảnh và lưu mỗi cảnh thành một file riêng.

        Args:
            video_path (str): Đường dẫn đến file video gốc cần xử lý.
            output_dir (str): Thư mục để lưu các file video của từng cảnh.
            video_id (str): ID định danh cho video gốc.

        Returns:
            List[Dict[str, Any]]: Một danh sách các dictionary, mỗi dictionary chứa thông tin
                                  về một cảnh đã được tách, bao gồm:
                                  - 'scene_path': Đường dẫn đến file video của cảnh.
                                  - 'scene_id': ID định danh duy nhất cho cảnh.
                                  - 'start_seconds': Thời điểm bắt đầu của cảnh trong video gốc.
                                  - 'duration_seconds': Thời lượng của cảnh.
        """
        if not os.path.exists(video_path):
            logger.error(f"File video không tồn tại: {video_path}")
            return []

        # Đảm bảo thư mục output tồn tại
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Bắt đầu phân cảnh cho video: {video_path}. Output sẽ được lưu tại: {output_dir}")

        try:
            # 1. Phát hiện các cảnh sử dụng ContentDetector
            # ContentDetector phát hiện các thay đổi đột ngột về nội dung hình ảnh.
            scene_list = detect(video_path, ContentDetector())

            if not scene_list:
                logger.warning(f"Không phát hiện được cảnh nào trong video '{video_id}'. Coi toàn bộ là một cảnh.")
                # Nếu không có cảnh, coi cả video là một cảnh duy nhất
                # (Trong thực tế có thể copy file gốc sang)
                return [{
                    "scene_path": video_path,
                    "scene_id": f"{video_id}_scene_0001",
                    "start_seconds": 0.0,
                    "duration_seconds": 0.0 # Cần tính toán lại nếu cần
                }]

            logger.info(f"Phát hiện được {len(scene_list)} cảnh. Bắt đầu tách thành các file...")

            # 2. Tách video thành các file riêng lẻ bằng ffmpeg
            # PySceneDetect cung cấp một hàm tiện ích để làm việc này
            split_video_ffmpeg(video_path, scene_list, output_dir=output_dir,
                               file_name_template=f'{video_id}_scene-$SCENE_NUMBER.mp4')
            
            # 3. Thu thập thông tin về các file cảnh đã được tạo
            processed_scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                scene_number = i + 1
                scene_file_name = f'{video_id}_scene-{scene_number:04d}.mp4'
                scene_file_path = os.path.join(output_dir, scene_file_name)

                if os.path.exists(scene_file_path):
                    scene_info = {
                        "scene_path": scene_file_path,
                        "scene_id": f"{video_id}_scene_{scene_number:04d}",
                        "start_seconds": start_time.get_seconds(),
                        "duration_seconds": (end_time - start_time).get_seconds()
                    }
                    processed_scenes.append(scene_info)
                else:
                    logger.warning(f"Không tìm thấy file cảnh dự kiến: {scene_file_path}")
            
            logger.info(f"Hoàn tất tách {len(processed_scenes)} cảnh cho video '{video_id}'.")
            return processed_scenes

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong quá trình phân cảnh video '{video_path}': {e}", exc_info=True)
            return []

