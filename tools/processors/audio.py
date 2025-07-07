# /tools/processors/audio.py
"""
Module trích xuất đặc trưng âm thanh (Audio Embeddings) từ video.
Sử dụng bộ mã hóa (encoder) của một mô hình Nhận dạng Giọng nói Tự động (ASR)
để tạo ra vector đặc trưng cho nội dung âm thanh.
"""
import logging
from typing import Optional

import torch
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)

# Cấu hình logging
logger = logging.getLogger(__name__)

class SotaAudioFeatureExtractor:
    """
    Trích xuất Audio Embeddings sử dụng encoder của một model ASR SOTA.
    
    Chúng ta không cần phần giải mã (decoder) để sinh ra văn bản, mà chỉ lấy
    trạng thái ẩn từ encoder làm vector đại diện cho âm thanh.
    """
    def __init__(self, model_id: str = "openai/whisper-base"):
        """
        Khởi tạo extractor.

        Args:
            model_id (str): ID của model ASR trên Hugging Face Hub.
                            'openai/whisper-base' là một lựa chọn tốt, cân bằng
                            giữa hiệu năng và kích thước.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        logger.info(f"SotaAudioFeatureExtractor khởi tạo với model '{self.model_id}' trên thiết bị '{self.device}'.")

    def _load_dependencies(self):
        """
        Tải model và processor từ Hugging Face Hub nếu chưa được tải.
        """
        if self._model is not None and self._processor is not None:
            return

        logger.info(f"Đang tải model Audio SOTA: {self.model_id}...")
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._model.eval()
            logger.info(f"Tải model '{self.model_id}' thành công lên thiết bị '{self.device}'.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi tải model Audio '{self.model_id}': {e}", exc_info=True)
            raise RuntimeError(f"Không thể tải model Audio: {e}") from e

    def get_embedding_from_audio(self, audio_array: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """
        Tạo vector embedding từ một mảng numpy chứa dữ liệu âm thanh.

        Args:
            audio_array (np.ndarray): Mảng numpy chứa waveform của audio.
            sample_rate (int): Tần số lấy mẫu của audio (ví dụ: 16000).

        Returns:
            Optional[np.ndarray]: Một vector numpy đại diện cho đoạn audio,
                                  hoặc None nếu có lỗi.
        """
        if audio_array is None or audio_array.size == 0:
            logger.warning("Mảng audio đầu vào rỗng. Bỏ qua việc tạo embedding.")
            return None
        
        self._load_dependencies()
        
        try:
            with torch.no_grad():
                # 1. Chuẩn bị đầu vào (mel spectrogram) cho model
                inputs = self._processor(
                    audio_array, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).to(self.device, torch.float16)
                
                # 2. Chỉ chạy qua bộ mã hóa (encoder) để lấy đặc trưng
                # Đây là phần cốt lõi của việc trích xuất embedding
                encoder_outputs = self._model.get_encoder()(**inputs)
                
                # 3. Lấy last_hidden_state từ encoder.
                # Kích thước sẽ là (batch_size, sequence_length, hidden_size)
                last_hidden_state = encoder_outputs.last_hidden_state
                
                # 4. Tính trung bình các vector theo chiều thời gian (sequence_length)
                # để có một vector duy nhất đại diện cho cả đoạn audio.
                audio_embedding = torch.mean(last_hidden_state, dim=1)
                
                # 5. Chuẩn hóa vector (L2 normalization)
                audio_embedding = audio_embedding / torch.linalg.norm(audio_embedding, ord=2, dim=-1, keepdim=True)
                
                logger.info("Đã tạo audio embedding từ encoder của model ASR.")
                
                # 6. Chuyển về CPU và trả về dưới dạng numpy array
                return audio_embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tạo audio embedding: {e}", exc_info=True)
            return None
