# /tools/video_processing/visual_analysis.py
"""
Tool cho visual analysis sử dụng các mô hình AI, tuân thủ kiến trúc Agno.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện xử lý ảnh và AI
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualAnalysisTool:
    """
    Một class chứa các công cụ để phân tích nội dung hình ảnh của keyframes.
    """
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large"):
        """
        Khởi tạo tool và các thuộc tính cần thiết để tải mô hình.
        """
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_dependencies(self):
        """
        Tải processor và model khi cần thiết (lazy loading).
        """
        if self._processor is None or self._model is None:
            logger.info(f"Đang tải model phân tích hình ảnh: '{self.model_id}' lên thiết bị '{self._device}'...")
            try:
                self._processor = BlipProcessor.from_pretrained(self.model_id)
                self._model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self._device)
                logger.info("Tải model thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi tải model '{self.model_id}': {e}", exc_info=True)
                raise RuntimeError(f"Không thể tải model Visual Analysis: {e}")

    @tool(
        name="analyze_image",
        description="Phân tích một hình ảnh để tạo mô tả (caption) và phát hiện đối tượng (chức năng giả định).",
        cache_results=True
    )
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Thực hiện phân tích toàn diện trên một hình ảnh.

        Args:
            image_path (str): Đường dẫn đến tệp hình ảnh (keyframe).

        Returns:
            Dict[str, Any]: Một dictionary chứa kết quả phân tích, bao gồm:
                            'description', 'objects', và 'embeddings' (giả định).
        """
        logger.info(f"Bắt đầu phân tích hình ảnh: {image_path}")
        self._load_dependencies() # Đảm bảo model đã được tải

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Tệp hình ảnh không tồn tại: {image_path}")

        try:
            # Tạo mô tả cho ảnh
            description = self._generate_caption(image_path)
            
            # Placeholder cho các chức năng phân tích khác
            objects = self.detect_objects(image_path) # Gọi tool khác trong cùng class
            embeddings = [0.1, 0.2, 0.3] # Placeholder

            return {
                "description": description,
                "objects": objects,
                "embeddings": embeddings
            }
        except Exception as e:
            logger.error(f"Lỗi khi phân tích hình ảnh '{image_path}': {e}", exc_info=True)
            raise e

    def _generate_caption(self, image_path: str) -> Optional[str]:
        """Hàm nội bộ để tạo caption, được gọi bởi tool `analyze_image`."""
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self._processor(images=raw_image, return_tensors="pt").to(self._device)
            output_ids = self._model.generate(**inputs, max_length=50)
            caption = self._processor.decode(output_ids[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.error(f"Lỗi phụ khi tạo caption: {e}")
            return "Không thể tạo mô tả cho hình ảnh này."


    @tool(
        name="detect_objects",
        description="Phát hiện các đối tượng trong một hình ảnh (chức năng giả định).",
        cache_results=True
    )
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Đây là một placeholder cho tool nhận dạng đối tượng.
        Trong một hệ thống thực tế, nó sẽ gọi một mô hình như YOLO hoặc DETR.

        Args:
            image_path (str): Đường dẫn đến tệp hình ảnh.

        Returns:
            List[Dict[str, Any]]: Danh sách các đối tượng được phát hiện.
        """
        logger.info(f"Đang thực hiện nhận dạng đối tượng giả định cho: {image_path}")
        # Logic giả định
        return [
            {"label": "person", "confidence": 0.95, "box": [10, 20, 50, 100]},
            {"label": "car", "confidence": 0.88, "box": [60, 40, 150, 120]}
        ]
