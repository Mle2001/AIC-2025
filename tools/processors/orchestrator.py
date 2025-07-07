# /tools/processors/orchestrator.py
"""
Module điều phối (Orchestrator).
Kết hợp tất cả các bộ xử lý (visual, audio, text) để thực hiện
một pipeline phân tích video đa phương thức hoàn chỉnh.
✅ ĐÃ CẬP NHẬT: Pipeline giờ đây xử lý các cảnh (scenes) một cách logic
trong bộ nhớ (in-memory) từ video gốc, thay vì tách thành các file vật lý.
Cách tiếp cận này hiệu quả hơn và tiết kiệm tài nguyên.
"""
import logging
import asyncio
from typing import Dict, Any, List, Tuple

import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import librosa
import noisereduce as nr

# PySceneDetect chỉ dùng để lấy danh sách mốc thời gian
from scenedetect import detect, ContentDetector

# Import các lớp xử lý từ các module khác
from .visual import SotaVisualFeatureExtractor
from .audio import SotaAudioFeatureExtractor
from .text import SotaOCRProcessor, TextFeatureExtractor

# Cấu hình logging
logger = logging.getLogger(__name__)


class VideoAnalysisOrchestrator:
    """
    Điều phối toàn bộ quy trình xử lý video:
    1. Phát hiện các mốc thời gian của từng cảnh trong video gốc.
    2. Với mỗi mốc thời gian, tạo một subclip ảo trong bộ nhớ.
    3. Trích xuất song song các đặc trưng đa phương thức (visual, audio, ocr) từ subclip ảo đó.
    4. Trả về một cấu trúc dữ liệu sạch, sẵn sàng để lưu vào Vector DB.
    """
    def __init__(self):
        """Khởi tạo tất cả các bộ xử lý con."""
        self.visual_extractor = SotaVisualFeatureExtractor()
        self.audio_extractor = SotaAudioFeatureExtractor()
        self.ocr_processor = SotaOCRProcessor()
        self.text_embedder = TextFeatureExtractor()
        logger.info("VideoAnalysisOrchestrator (SOTA) đã được khởi tạo với đầy đủ các bộ xử lý.")

    def _get_scene_timestamps(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Phân tích video và chỉ trả về danh sách các mốc thời gian của cảnh.
        Đây là một hàm tiện ích nội bộ, không phải là một "Agno tool".

        Args:
            video_path (str): Đường dẫn đến file video gốc.

        Returns:
            List[Tuple[float, float]]: Danh sách các cảnh, mỗi cảnh là một tuple
                                       (start_time_seconds, end_time_seconds).
        """
        try:
            # Sử dụng ContentDetector để tìm các điểm chuyển cảnh
            scene_list_frames = detect(video_path, ContentDetector())
            
            if not scene_list_frames:
                logger.warning("Không phát hiện được cảnh nào. Coi toàn bộ video là một cảnh duy nhất.")
                with VideoFileClip(video_path) as clip:
                    return [(0.0, clip.duration)]

            # Chuyển đổi từ FrameTimecode sang giây
            scene_list_seconds = [(s.get_seconds(), e.get_seconds()) for s, e in scene_list_frames]
            logger.info(f"Phát hiện được {len(scene_list_seconds)} cảnh.")
            return scene_list_seconds
        except Exception as e:
            logger.error(f"Lỗi trong quá trình phát hiện mốc thời gian cảnh: {e}", exc_info=True)
            return []

    def _extract_data_from_subclip(self, scene_clip: VideoFileClip, frames_per_scene=16, ocr_frame_time=0.5) -> Dict[str, Any]:
        """
        Trích xuất dữ liệu thô (khung hình, audio, keyframe) từ một đối tượng subclip trong bộ nhớ.

        Args:
            scene_clip (VideoFileClip): Đối tượng subclip của MoviePy.
            frames_per_scene (int): Số lượng khung hình cần trích xuất.
            ocr_frame_time (float): Vị trí tương đối trong cảnh để lấy khung hình cho OCR.

        Returns:
            Dict[str, Any]: Dictionary chứa dữ liệu thô.
        """
        duration = scene_clip.duration
        if duration <= 0:
            return {"frames": [], "audio_array": None, "sample_rate": 16000, "ocr_frame": None}

        # 1. Trích xuất các khung hình cho visual embedding
        frames = [Image.fromarray(scene_clip.get_frame(t)) for t in np.linspace(0, duration, num=frames_per_scene, endpoint=False)]

        # 2. Trích xuất và tiền xử lý audio
        audio_array, sample_rate = None, 16000
        if scene_clip.audio:
            try:
                audio_array_raw = scene_clip.audio.to_soundarray(fps=sample_rate)
                audio_array_mono = audio_array_raw.mean(axis=1) if audio_array_raw.ndim > 1 else audio_array_raw
                audio_array = nr.reduce_noise(y=audio_array_mono, sr=sample_rate)
            except Exception as e:
                logger.warning(f"Không thể xử lý audio cho cảnh: {e}")
                audio_array = None

        # 3. Lấy một khung hình đại diện cho OCR
        key_frame_for_ocr = Image.fromarray(scene_clip.get_frame(duration * ocr_frame_time))

        return {"frames": frames, "audio_array": audio_array, "sample_rate": sample_rate, "ocr_frame": key_frame_for_ocr}

    async def process_video_by_scenes(self, video_path: str, video_id: str) -> List[Dict[str, Any]]:
        """
        Pipeline xử lý chính, hoạt động trên các cảnh logic.

        Args:
            video_path (str): Đường dẫn đến video gốc cần xử lý.
            video_id (str): Một ID định danh cho video này.

        Returns:
            List[Dict[str, Any]]: Danh sách các dictionary, mỗi cái đại diện cho một cảnh
                                  đã được xử lý và sẵn sàng để lưu trữ.
        """
        logger.info(f"Bắt đầu xử lý video '{video_path}' theo từng cảnh logic.")
        
        # Bước 1: Lấy danh sách các mốc thời gian của cảnh
        scene_timestamps = await asyncio.to_thread(self._get_scene_timestamps, video_path)
        if not scene_timestamps:
            return []

        # Mở file video gốc một lần duy nhất
        with VideoFileClip(video_path) as full_video_clip:
            results_for_db = []
            for i, (start, end) in enumerate(scene_timestamps):
                scene_id = f"{video_id}_scene_{i+1:04d}"
                logger.info(f"Đang xử lý cảnh {i+1}/{len(scene_timestamps)} (từ {start:.2f}s đến {end:.2f}s)...")
                
                # Bước 2: Tạo một subclip ảo trong bộ nhớ cho cảnh hiện tại
                scene_clip = full_video_clip.subclip(start, end)
                
                # Bước 3: Trích xuất dữ liệu thô từ subclip ảo
                scene_data = await asyncio.to_thread(self._extract_data_from_subclip, scene_clip)

                # Bước 4: Chạy các tác vụ trích xuất embedding nặng song song
                tasks = {
                    "visual": asyncio.to_thread(self.visual_extractor.get_embedding_from_frames, scene_data["frames"]),
                    "audio": asyncio.to_thread(self.audio_extractor.get_embedding_from_audio, scene_data["audio_array"], scene_data["sample_rate"]),
                    "ocr": asyncio.to_thread(self.ocr_processor.extract_text_from_frame, scene_data["ocr_frame"])
                }
                
                task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                # Gán kết quả và xử lý lỗi nếu có
                visual_embedding = task_results[0] if not isinstance(task_results[0], Exception) else None
                audio_embedding = task_results[1] if not isinstance(task_results[1], Exception) else None
                ocr_result = task_results[2] if not isinstance(task_results[2], Exception) else {"full_text": ""}

                # Tạo embedding cho text từ OCR (nếu có)
                ocr_text = ocr_result.get("full_text", "")
                ocr_embedding = None
                if ocr_text:
                    ocr_embedding = await asyncio.to_thread(self.text_embedder.get_embedding_from_text, ocr_text)

                # Bước 5: Tổng hợp kết quả cho cảnh này
                scene_record = {
                    "scene_id": scene_id,
                    "video_id": video_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "visual_embedding": visual_embedding.flatten().tolist() if visual_embedding is not None else None,
                    "audio_embedding": audio_embedding.flatten().tolist() if audio_embedding is not None else None,
                    "ocr_embedding": ocr_embedding.flatten().tolist() if ocr_embedding is not None else None,
                    "raw_ocr_text": ocr_text
                }
                results_for_db.append(scene_record)

        logger.info(f"Hoàn tất xử lý, trích xuất được {len(results_for_db)} cụm thông tin (cảnh) từ video '{video_id}'.")
        return results_for_db
