# /tools/processors/visual.py
"""
Module trích xuất đặc trưng hình ảnh (Visual Embeddings) từ video.
Sử dụng mô hình Video-LLaVA để tạo ra các vector đặc trưng giàu ngữ nghĩa.
"""
import logging
from typing import List, Optional

import torch
import numpy as np
from PIL import Image
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor
)

# Cấu hình logging
logger = logging.getLogger(__name__)

class SotaVisualFeatureExtractor:
    """
    Trích xuất Visual Embeddings sử dụng Video-LLaVA.
    
    Thay vì sinh văn bản, chúng ta trích xuất trạng thái ẩn cuối cùng từ language model
    sau khi nó đã "nhìn thấy" các khung hình. Vector này đại diện cho "thông tin trung gian"
    về mặt hình ảnh.
    """
    def __init__(self, model_id: str = "PKU-YuanGroup/Video-LLaVA-7B"):
        """
        Khởi tạo extractor.

        Args:
            model_id (str): ID của model trên Hugging Face Hub.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        logger.info(f"SotaVisualFeatureExtractor khởi tạo với model '{self.model_id}' trên thiết bị '{self.device}'.")

    def _load_dependencies(self):
        """
        Tải model và processor từ Hugging Face Hub nếu chưa được tải.
        Sử dụng lazy loading để tiết kiệm bộ nhớ khi khởi tạo.
        """
        if self._model is not None and self._processor is not None:
            return

        logger.info(f"Đang tải model Video-Language SOTA: {self.model_id}...")
        try:
            self._processor = VideoLlavaProcessor.from_pretrained(self.model_id)
            self._model = VideoLlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._model.eval() # Chuyển model sang chế độ đánh giá (inference)
            logger.info(f"Tải model '{self.model_id}' thành công lên thiết bị '{self.device}'.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi tải model '{self.model_id}': {e}", exc_info=True)
            raise RuntimeError(f"Không thể tải model Video-LLaVA: {e}") from e

    def get_embedding_from_frames(self, frames: List[Image.Image]) -> Optional[np.ndarray]:
        """
        Tạo vector embedding từ một danh sách các khung hình (đại diện cho một cảnh).

        Args:
            frames (List[Image.Image]): Danh sách các đối tượng PIL Image.

        Returns:
            Optional[np.ndarray]: Một vector numpy đại diện cho cảnh,
                                  hoặc None nếu có lỗi xảy ra.
        """
        if not frames:
            logger.warning("Danh sách khung hình đầu vào rỗng. Bỏ qua việc tạo embedding.")
            return None
        
        self._load_dependencies()
        
        # Một prompt chung để hướng dẫn model tập trung vào việc mô tả nội dung hình ảnh.
        prompt = "A comprehensive description of the video scene is:"
        
        try:
            with torch.no_grad():
                # 1. Chuẩn bị đầu vào cho model
                inputs = self._processor(
                    text=prompt, 
                    images=frames, 
                    return_tensors="pt"
                ).to(self.device, torch.float16)
                
                # 2. Chạy forward pass để lấy hidden states, không cần generate text
                # output_hidden_states=True là chìa khóa để truy cập các trạng thái ẩn
                outputs = self._model(**inputs, output_hidden_states=True)
                
                # 3. Trích xuất hidden states từ tầng cuối cùng của language model
                # outputs.hidden_states là một tuple, phần tử cuối (-1) là của tầng cuối cùng
                last_hidden_states = outputs.hidden_states[-1]
                
                # 4. Lấy vector của token cuối cùng làm đại diện cho cả cảnh.
                # Đây là một kỹ thuật phổ biến để có được embedding đại diện cho cả chuỗi.
                scene_embedding = last_hidden_states[:, -1, :]
                
                # 5. Chuẩn hóa vector (L2 normalization). Bước này rất quan trọng
                # để đảm bảo các vector có cùng độ lớn, giúp việc so sánh cosine hiệu quả hơn.
                scene_embedding = scene_embedding / torch.linalg.norm(scene_embedding, ord=2, dim=-1, keepdim=True)
                
                logger.info(f"Đã tạo visual embedding từ Video-LLaVA cho {len(frames)} khung hình.")
                
                # 6. Chuyển embedding về CPU và trả về dưới dạng numpy array
                return scene_embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tạo visual embedding với Video-LLaVA: {e}", exc_info=True)
            return None

