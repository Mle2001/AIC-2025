# /tools/processors/text.py
"""
Module xử lý văn bản, bao gồm Nhận dạng Ký tự Quang học (OCR)
và tạo vector đặc trưng cho văn bản (Text Embedding).
"""
import logging
from typing import Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer

# Cấu hình logging
logger = logging.getLogger(__name__)


class SotaOCRProcessor:
    """
    Sử dụng Florence-2, một mô hình Vision-Language mạnh mẽ, để thực hiện OCR.
    Trả về cả chuỗi văn bản đầy đủ và vị trí của chúng.
    """
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        """
        Khởi tạo processor.

        Args:
            model_id (str): ID của model trên Hugging Face Hub.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        logger.info(f"SotaOCRProcessor khởi tạo với model '{self.model_id}' trên thiết bị '{self.device}'.")

    def _load_dependencies(self):
        """
        Tải model và processor, chỉ khi cần thiết.
        Florence-2 yêu cầu `trust_remote_code=True`.
        """
        if self._model is not None and self._processor is not None:
            return

        logger.info(f"Đang tải model OCR SOTA: {self.model_id}...")
        try:
            # Florence-2 yêu cầu trust_remote_code=True
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            self._model.eval()
            logger.info(f"Tải model OCR '{self.model_id}' thành công.")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi tải model OCR '{self.model_id}': {e}", exc_info=True)
            raise RuntimeError(f"Không thể tải model OCR: {e}") from e

    def extract_text_from_frame(self, frame: Image.Image) -> Dict[str, Any]:
        """
        Trích xuất văn bản từ một khung hình (PIL Image).

        Args:
            frame (Image.Image): Khung hình cần phân tích.

        Returns:
            Dict[str, Any]: Một dictionary chứa 'full_text' và 'regions' (văn bản và vị trí).
        """
        if frame is None:
            return {"full_text": "", "regions": []}
            
        self._load_dependencies()
        
        # Prompt đặc biệt để kích hoạt tác vụ OCR của Florence-2
        prompt = "<OCR_WITH_REGION>"
        
        try:
            # 1. Chuẩn bị đầu vào
            inputs = self._processor(
                text=prompt, 
                images=frame, 
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            # 2. Sinh ra kết quả dưới dạng token
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3, # Sử dụng beam search để có kết quả tốt hơn
                )
            
            # 3. Decode token thành văn bản
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # 4. Hậu xử lý để có được kết quả OCR có cấu trúc
            parsed_results = self._processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=frame.size
            )
            
            ocr_result = parsed_results.get(prompt, {"labels": [], "quad_boxes": []})
            full_text = " ".join(ocr_result["labels"])
            logger.info(f"OCR hoàn tất, tìm thấy text: '{full_text[:100]}...'")
            
            return {
                "full_text": full_text,
                "regions": [{"text": label, "box": box} for label, box in zip(ocr_result["labels"], ocr_result["quad_boxes"])]
            }
        except Exception as e:
            logger.error(f"Lỗi trong quá trình OCR với Florence-2: {e}", exc_info=True)
            return {"full_text": "", "regions": []}


class TextFeatureExtractor:
    """
    Chuyển đổi một chuỗi văn bản thành một vector embedding.
    Lớp này rất quan trọng cho cả việc mã hóa văn bản từ OCR và mã hóa
    truy vấn của người dùng để thực hiện tìm kiếm.
    """
    def __init__(self, model_id: str = 'all-MiniLM-L6-v2'):
        """
        Khởi tạo extractor.

        Args:
            model_id (str): ID của model SentenceTransformer. 'all-MiniLM-L6-v2'
                            nhanh, nhẹ và hiệu quả cho tìm kiếm ngữ nghĩa.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        logger.info(f"TextFeatureExtractor khởi tạo với model '{self.model_id}' trên thiết bị '{self.device}'.")

    def _load_dependencies(self):
        """
        Tải model SentenceTransformer.
        """
        if self._model is not None:
            return
            
        logger.info(f"Đang tải model Text Embedding: {self.model_id}...")
        try:
            self._model = SentenceTransformer(self.model_id, device=self.device)
            logger.info("Tải model Text Embedding thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi tải model Text Embedding '{self.model_id}': {e}", exc_info=True)
            raise RuntimeError(f"Không thể tải model Text Embedding: {e}") from e
    
    def get_embedding_from_text(self, text: str) -> Optional[np.ndarray]:
        """
        Tạo vector embedding từ một chuỗi văn bản.

        Args:
            text (str): Chuỗi văn bản cần mã hóa.

        Returns:
            Optional[np.ndarray]: Vector embedding, hoặc None nếu text rỗng.
        """
        if not text or not text.strip():
            return None
            
        self._load_dependencies()
        try:
            # normalize_embeddings=True sẽ tự động chuẩn hóa L2 cho vector
            embedding = self._model.encode(
                [text], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            logger.info(f"Đã tạo text embedding cho: '{text[:50]}...'")
            return embedding
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tạo text embedding: {e}", exc_info=True)
            return None

