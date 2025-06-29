# /tools/video_processing/keyframe_extraction.py
"""
Tool cho keyframe extraction từ các video scenes, tuân thủ kiến trúc Agno.
Công cụ này được duy trì để sử dụng song song hoặc làm phương án dự phòng.
"""
import cv2
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeyframeExtractionTool:
    """
    Một class chứa các công cụ liên quan đến việc trích xuất khung hình chính (keyframes).
    """

    @tool(
        name="extract_keyframes",
        description="Trích xuất các khung hình đại diện từ các phân đoạn video đã cho.",
        cache_results=True
    )
    def extract_keyframes(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
        output_dir: str = "./artifacts/keyframes"
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất keyframe từ mỗi cảnh trong danh sách bằng cách lấy khung hình ở giữa.

        Args:
            video_path (str): Đường dẫn đến tệp video.
            scenes (List[Dict]): Danh sách các cảnh, mỗi cảnh là một dict có 'start_time' và 'end_time'.
            output_dir (str): Thư mục để lưu các tệp ảnh keyframe.

        Returns:
            List[Dict[str, Any]]: Danh sách các keyframe, chứa 'timestamp', 'image_path', và 'scene_index'.
        """
        logger.info(f"Bắt đầu trích xuất keyframes cho {len(scenes)} cảnh từ video: {video_path}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            logger.error(f"Không thể mở video: {video_path}")
            raise IOError(f"Could not open video file: {video_path}")

        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            vid_cap.release()
            raise ValueError(f"Video file {video_path} has an invalid FPS of 0.")

        extracted_keyframes = []
        try:
            for i, scene in enumerate(scenes):
                start_time = scene.get("start_time")
                end_time = scene.get("end_time")

                if start_time is None or end_time is None:
                    logger.warning(f"Bỏ qua cảnh thứ {i} vì thiếu 'start_time' hoặc 'end_time'.")
                    continue

                middle_time = start_time + (end_time - start_time) / 2
                frame_position = int(middle_time * fps)
                
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                success, frame = vid_cap.read()

                if success:
                    video_name = Path(video_path).stem
                    keyframe_filename = f"{video_name}_scene_{i:03d}_time_{middle_time:.2f}.jpg".replace('.', '_')
                    image_path = os.path.join(output_dir, keyframe_filename)
                    
                    cv2.imwrite(image_path, frame)
                    
                    keyframe_data = {
                        "timestamp": middle_time,
                        "image_path": image_path,
                        "scene_index": i,
                        "metadata": {
                            "source_video": video_path,
                            "scene_start": start_time,
                            "scene_end": end_time
                        }
                    }
                    extracted_keyframes.append(keyframe_data)
                    logger.info(f"Đã trích xuất keyframe cho cảnh {i} tại {middle_time:.2f}s")
                else:
                    logger.warning(f"Không thể đọc khung hình cho cảnh {i} tại thời điểm {middle_time:.2f}s.")
        
        finally:
            vid_cap.release()
            
        logger.info(f"Hoàn tất trích xuất, tổng cộng {len(extracted_keyframes)} keyframes.")
        return extracted_keyframes
