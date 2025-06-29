# /tools/video_processing/visual_analysis.py
"""
Tool cho visual analysis sử dụng các mô hình như BLIP và DETR, tuân thủ kiến trúc Agno.
Công cụ này được duy trì để sử dụng song song hoặc làm phương án dự phòng.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List

# Thư viện Agno để định nghĩa tool
from agno.tools import tool

# Thư viện xử lý ảnh và AI
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    DetrImageProcessor, DetrForObjectDetection
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualAnalysisTool:
    """
    Một class chứa các công cụ để phân tích nội dung hình ảnh của keyframes.
    - Tạo mô tả (Captioning) bằng BLIP.
    - Nhận dạng đối tượng (Object Detection) bằng DETR.
    """
    def __init__(
        self, 
        captioning_model_id: str = "Salesforce/blip-image-captioning-large",
        detection_model_id: str = "facebook/detr-resnet-50"
    ):
        """
        Khởi tạo tool và các thuộc tính cần thiết để tải mô hình.
        """
        self.captioning_model_id = captioning_model_id
        self.detection_model_id = detection_model_id
        
        self._caption_processor = None
        self._caption_model = None
        self._detection_processor = None
        self._detection_model = None
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"VisualAnalysisTool (Legacy) khởi tạo trên thiết bị '{self._device}'.")

    def _load_captioning_dependencies(self):
        """Tải processor và model cho việc tạo mô tả (BLIP) khi cần."""
        if self._caption_processor is None or self._caption_model is None:
            logger.info(f"Đang tải model BLIP (Legacy): '{self.captioning_model_id}'...")
            try:
                self._caption_processor = BlipProcessor.from_pretrained(self.captioning_model_id)
                self._caption_model = BlipForConditionalGeneration.from_pretrained(self.captioning_model_id).to(self._device)
                logger.info("Tải model BLIP thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi tải model BLIP '{self.captioning_model_id}': {e}", exc_info=True)
                raise RuntimeError(f"Không thể tải model Captioning (Legacy): {e}")

    def _load_detection_dependencies(self):
        """Tải processor và model cho việc nhận dạng đối tượng (DETR) khi cần."""
        if self._detection_processor is None or self._detection_model is None:
            logger.info(f"Đang tải model DETR (Legacy): '{self.detection_model_id}'...")
            try:
                self._detection_processor = DetrImageProcessor.from_pretrained(self.detection_model_id)
                self._detection_model = DetrForObjectDetection.from_pretrained(self.detection_model_id).to(self._device)
                logger.info("Tải model DETR thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi tải model DETR '{self.detection_model_id}': {e}", exc_info=True)
                raise RuntimeError(f"Không thể tải model Detection (Legacy): {e}")

    @tool(
        name="analyze_image_legacy",
        description="Phân tích toàn diện một hình ảnh: tạo mô tả (caption) và nhận dạng đối tượng.",
        cache_results=True
    )
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Thực hiện phân tích trên một hình ảnh để tạo mô tả và danh sách đối tượng.

        Args:
            image_path (str): Đường dẫn đến tệp hình ảnh (keyframe).

        Returns:
            Dict[str, Any]: Một dictionary chứa 'description' và 'objects'.
        """
        logger.info(f"Bắt đầu phân tích hình ảnh toàn diện (Legacy) cho: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Tệp hình ảnh không tồn tại: {image_path}")

        try:
            # Tải và mở ảnh một lần duy nhất
            raw_image = Image.open(image_path).convert('RGB')
            
            # Tạo mô tả
            caption = self._generate_caption(raw_image)
            
            # Nhận dạng đối tượng
            objects = self.detect_objects(image_path, preloaded_image=raw_image)
            
            return {
                "description": caption,
                "objects": objects
            }
        except Exception as e:
            logger.error(f"Lỗi khi phân tích hình ảnh '{image_path}': {e}", exc_info=True)
            raise e

    def _generate_caption(self, image: Image.Image) -> str:
        """Hàm nội bộ để tạo caption, nhận vào đối tượng ảnh PIL."""
        self._load_captioning_dependencies()
        inputs = self._caption_processor(images=image, return_tensors="pt").to(self._device)
        output_ids = self._caption_model.generate(**inputs, max_length=50)
        caption = self._caption_processor.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Đã tạo caption (Legacy): '{caption}'")
        return caption.strip()

    @tool(
        name="detect_objects_legacy",
        description="Phát hiện và định vị các đối tượng trong một hình ảnh bằng mô hình DETR.",
        cache_results=True
    )
    def detect_objects(self, image_path: str, preloaded_image: Image.Image = None) -> List[Dict[str, Any]]:
        """
        Sử dụng DETR để nhận dạng đối tượng trong ảnh.

        Args:
            image_path (str): Đường dẫn đến tệp ảnh (để ghi log).
            preloaded_image (Image.Image, optional): Đối tượng ảnh PIL đã được tải trước. 
                                                     Nếu không có, ảnh sẽ được tải từ image_path.

        Returns:
            List[Dict[str, Any]]: Danh sách các đối tượng, mỗi đối tượng là một dict
                                  chứa 'label', 'confidence', và 'box'.
        """
        logger.info(f"Bắt đầu nhận dạng đối tượng (Legacy) cho: {image_path}")
        self._load_detection_dependencies()
        
        if preloaded_image:
            image = preloaded_image
        else:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Tệp hình ảnh không tồn tại: {image_path}")
            image = Image.open(image_path).convert('RGB')

        inputs = self._detection_processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._detection_model(**inputs)

        # Chuyển đổi kết quả đầu ra thành định dạng mong muốn
        target_sizes = torch.tensor([image.size[::-1]]).to(self._device)
        results = self._detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detected_objects.append({
                "label": self._detection_model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "box": box
            })
        
        logger.info(f"Đã phát hiện {len(detected_objects)} đối tượng.")
        return detected_objects
