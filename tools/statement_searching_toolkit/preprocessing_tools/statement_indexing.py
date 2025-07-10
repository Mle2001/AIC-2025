# ============================
# 📦 IMPORTS - Giải thích mục đích từng gói
# ============================
from typing import List, Optional  # 👉 Kiểu dữ liệu cho type hinting (List, Optional)
from agno.tools import tool        # 👉 Decorator để đăng ký tool cho agno framework
import logging                     # 👉 Ghi log cho quá trình xử lý, debug
import chromadb                    # 👉 Thư viện thao tác với vector database Chroma
from chromadb.config import Settings  # 👉 Cấu hình cho ChromaDB (ví dụ: đường dẫn lưu trữ)
import re                          # 👉 Xử lý chuỗi, tách câu, tách object bằng regex
from sentence_transformers import SentenceTransformer  # 👉 Model embedding câu (sentence-transformers)
import numpy as np                 # 👉 Xử lý mảng số, chuyển đổi embedding sang numpy array
import torch                     # 👉 Sử dụng PyTorch cho mô hình Video-LLaVA
from PIL import Image # 👉 Xử lý ảnh (PIL Image) để tương thích với mô hình Video-LLaVA
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration  # 👉 Sử dụng mô hình Video-LLaVA (LlavaNext) từ transformers
from transformers import AutoTokenizer # 👉 Tokenizer cho mô hình Video-LLaVA (LlavaNext)
import logging                   # 👉 Ghi log cho quá trình xử lý, debug
import re                         # 👉 Xử lý chuỗi, tách câu, tách object bằng regex
import numpy as np               # 👉 Xử lý mảng số, chuyển đổi embedding sang numpy array
from typing import List, Optional   # 👉 Kiểu dữ liệu cho type hinting (List, Optional)
from PIL import Image # 👉 Xử lý ảnh (PIL Image) để tương thích với mô hình Video-LLaVA
from sentence_transformers import SentenceTransformer # 👉 Model embedding câu (sentence-transformers)
from chromadb.config import Settings # 👉 Cấu hình cho ChromaDB (ví dụ: đường dẫn lưu trữ)
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration # 👉 Sử dụng mô hình Video-LLaVA (LlavaNext) từ transformers

"""
Pipeline for extracting, enriching, embedding, and storing statements from video frames using LLaVA-NeXT and sentence-transformers.
Code style: professional, concise, consistent, with clear comments and separation of imports and logic.
"""

# ============================
# 📦 IMPORTS - All at top as per code standards
# ============================


class FrameInfo:
    """
    � Đối tượng chứa thông tin về một frame hình ảnh, bao gồm ảnh, id frame, id video, id scene.
    """
    def __init__(self, frame, frame_id: str, video_id: str, scene_id: Optional[str] = None):
        self.frame = frame  # Ảnh (PIL.Image)
        self.frame_id = frame_id
        self.video_id = video_id
        self.scene_id = scene_id

class StatementItem:
    """
    📝 Đối tượng chứa một statement (phát biểu) trích xuất từ frame, kèm metadata cơ bản.
    """
    def __init__(self, text: str, frame_id: str, video_id: str, scene_id: Optional[str]):
        self.text = text
        self.frame_id = frame_id
        self.video_id = video_id
        self.scene_id = scene_id


class EmbeddedStatement:
    """
    🧩 Đối tượng chứa statement đã embedding (vector hóa), bao gồm metadata và vector embedding.
    """
    def __init__(self, text: str, frame_id: str, video_id: str, scene_id: Optional[str], embedding: List[float]):
        self.text = text
        self.frame_id = frame_id
        self.video_id = video_id
        self.scene_id = scene_id
        self.embedding = embedding


class StatementIndexingTools:
    """
    📝 Bộ công cụ tiền xử lý cho indexing statement (phát biểu, câu, đoạn) theo kiến trúc agno.
    Dùng để trích xuất, enrich, embedding và lưu trữ các statement từ frame hình ảnh vào vector database.
    """
    def __init__(self, video_llava_model_id: str = "lmms-lab/LLaVA-Video-7B-Qwen2", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Khởi tạo tool, logger, và embedding model. Không sử dụng VideoLanguageProcessor, tự implement pipeline Video-LLaVA.
        Args:
            video_llava_model_id: Model id cho Video-LLaVA (mặc định: lmms-lab/LLaVA-Video-7B-Qwen2, model thật trên HuggingFace)
            embedding_model_name: Tên model embedding cho sentence-transformers
        """
        self.logger = logging.getLogger(__name__)
        self.video_llava_model_id = video_llava_model_id  # Lưu model_id để dùng cho pipeline
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Tự động chọn thiết bị
        self.llava_processor = LlavaNextProcessor.from_pretrained(video_llava_model_id)
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(video_llava_model_id)
        self.llava_model = self.llava_model.to(self.device)
        # LLaVA-NeXT: tokenizer phải load riêng

        self.llava_tokenizer = AutoTokenizer.from_pretrained(video_llava_model_id)
        # 🧠 Khởi tạo sentence-transformers model một lần duy nhất
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def _run_llava_inference(self, frame, prompt, max_new_tokens=128):
        """
        🧠 Hàm nội bộ chạy inference Video-LLaVA cho một frame và prompt.
        Args:
            frame: Ảnh đầu vào (numpy array hoặc PIL Image)
            prompt: Prompt dạng text cho model
            max_new_tokens: Số token sinh tối đa
        Returns:
            response: Chuỗi text trả về từ model
        """

        # Chuyển frame sang PIL Image nếu cần
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        # Tokenize prompt
        input_ids = self.llava_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        # Process image (LlavaNextProcessor has .image_processor attribute)
        image_tensor = self.llava_processor.image_processor(frame, return_tensors="pt").pixel_values.to(self.device)
        inputs = {
            'input_ids': input_ids,
            'pixel_values': image_tensor
        }
        with self.torch.no_grad():
            generated_ids = self.llava_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            response = self.llava_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
       
    @tool(
        name="embed_statements",
        description="Nhận vào danh sách statement (list[StatementItem]), trả về list EmbeddedStatement với embedding vector tương ứng.",
        cache_results=False
    )
    def embed_statements(self, items: List[StatementItem], model_name: Optional[str] = None) -> List[EmbeddedStatement]:
        """
        Nhúng các StatementItem thành vector embedding bằng sentence-transformers, trả về list EmbeddedStatement.
        Args:
            items: List[StatementItem] - Danh sách statement cần embedding
            model_name: Tên model embedding (nếu None sẽ dùng model đã khởi tạo trong __init__)
        Returns:
            List[EmbeddedStatement] - Danh sách statement đã embedding (bao gồm metadata và vector)
        """
        model = self.embedding_model if model_name is None or model_name == self.embedding_model_name else SentenceTransformer(model_name)  # 🧠 Dùng model đã khởi tạo nếu không truyền model_name
        self.logger.info(f"Embedding {len(items)} statements bằng model: {model_name or self.embedding_model_name}")
        try:
            texts = [item.text for item in items]
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            embedded_items = [
                EmbeddedStatement(
                    text=item.text,
                    frame_id=item.frame_id,
                    video_id=item.video_id,
                    scene_id=item.scene_id,
                    embedding=embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i])
                )
                for i, item in enumerate(items)
            ]
            return embedded_items
        except Exception as e:
            self.logger.error(f"Lỗi khi embedding statements: {e}")
            return []

    @tool(
        name="save_embeddings_to_vectordb",
        description="Lưu embedding vectors vào vector database Chroma. Nhận vào list EmbeddedStatement (bao gồm embedding, text, metadata).",
        cache_results=False
    )
    def save_embeddings_to_vectordb(self, embedded_items: List[EmbeddedStatement], collection_name: str = "statements") -> bool:
        """
        Lưu danh sách EmbeddedStatement (bao gồm embedding vector và metadata) vào vector database Chroma.
        Args:
            embedded_items: List[EmbeddedStatement] - Danh sách statement đã embedding
            collection_name: Tên collection trong Chroma
        Returns:
            True nếu lưu thành công, False nếu có lỗi
        """
        self.logger.info(f"Lưu {len(embedded_items)} embedding vào Chroma collection: {collection_name}")
        try:
            client = chromadb.Client(Settings(persist_directory="./chroma_data"))
            # Tạo collection nếu chưa có
            if collection_name not in [c.name for c in client.list_collections()]:
                collection = client.create_collection(collection_name)
            else:
                collection = client.get_collection(collection_name)
            # Chuẩn bị dữ liệu
            ids = [f"{item.video_id}_{item.frame_id}_{i}" for i, item in enumerate(embedded_items)]
            texts = [item.text for item in embedded_items]
            embeddings = [item.embedding for item in embedded_items]
            np_embeddings = np.array(embeddings, dtype=np.float32)

            # 📝 Chuẩn bị metadata cho từng embedding
         
            metadatas = [
                {
                    "frame_id": str(item.frame_id) if item.frame_id is not None else None,
                    "video_id": str(item.video_id) if item.video_id is not None else None,
                    "scene_id": str(item.scene_id) if item.scene_id is not None else None,
                    "text": str(item.text) if item.text is not None else None
                }
                for item in embedded_items
            ]  # type: ignore  # ⚡ Bypass type checker for Chroma metadatas

            # Lưu vào Chroma, bao gồm cả metadata
            safe_metadatas = [
                {k: (v if v is not None else "") for k, v in meta.items()}
                for meta in metadatas
            ]
            collection.add(
                ids=ids,
                embeddings=np_embeddings,
                documents=texts,
                metadatas=safe_metadatas  # type: ignore
            )
            self.logger.info("Lưu embedding thành công vào Chroma.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu embedding vào Chroma: {e}")
            return False
 
    @tool(
        name="extract_statements_from_frame",
        description="Nhận vào một frame hình ảnh, trích xuất danh sách object, enrich object (gom nhóm), sinh nhiều statement cho mỗi object (mỗi khía cạnh), và sinh statement về quan hệ giữa các object (bao gồm cả nhóm) bằng Video-LLaVA.",
        cache_results=False
    )
    async def extract_statements_from_frame(self, frame_info: FrameInfo) -> List[StatementItem]:
        """
        📸 Nhận vào một FrameInfo (chứa ảnh, frame_id, video_id, scene_id), trích xuất danh sách object, enrich object (gom nhóm), sinh nhiều statement cho mỗi object (mỗi khía cạnh), và sinh statement về quan hệ giữa các object (bao gồm cả nhóm) bằng Video-LLaVA.
        Args:
            frame_info: FrameInfo - Thông tin về frame hình ảnh
        Returns:
            List[StatementItem] - Danh sách statement đã trích xuất (bao gồm metadata)
        """
        frame = frame_info.frame
        frame_id = frame_info.frame_id
        video_id = frame_info.video_id
        scene_id = frame_info.scene_id

        # Bước 1: Trích xuất danh sách object bằng hàm nội bộ _run_llava_inference
        prompt_objects = "Liệt kê tất cả các đối tượng xuất hiện trong ảnh, trả về dưới dạng danh sách, không giải thích."
        objects_text = self._run_llava_inference(frame, prompt_objects)

        objects = []
        if '[' in objects_text and ']' in objects_text:
            try:
                objects = eval(objects_text)
                if not isinstance(objects, list):
                    objects = re.split(r',\s*', objects_text.strip("[] "))
            except Exception:
                objects = re.split(r',\s*', objects_text.strip("[] "))
        else:
            objects = re.split(r',\s*', objects_text)
        objects = [o.strip('"\' \\') for o in objects if o.strip()]
        self.logger.info(f"Đối tượng phát hiện: {objects}")

        # Bước 2: Sinh nhiều statement cho từng object (mỗi statement mô tả một khía cạnh khác nhau)
        statements = []
        for obj in objects:
            # 📝 Prompt: yêu cầu trả về nhiều câu, mỗi câu mô tả một khía cạnh khác nhau của object
            prompt_obj_stmt = (
                f"Liệt kê nhiều câu mô tả ngắn gọn về đối tượng '{obj}' trong ảnh này, "
                f"mỗi câu tập trung vào một khía cạnh khác nhau như: đặc điểm nhận dạng, trạng thái, hành động, vị trí trong khung hình, màu sắc, kích thước, thuộc tính nổi bật... "
                f"Chỉ trả về danh sách các câu, không giải thích thêm."
            )
            obj_desc_text = self._run_llava_inference(frame, prompt_obj_stmt).strip()  # 🚀 Gọi hàm nội bộ thay vì await processor
            # Tách các câu mô tả (mỗi câu là một statement riêng)
            obj_descs = re.split(r'(?<=[.!?])\s+', obj_desc_text)
            obj_descs = [desc for desc in obj_descs if len(desc.strip()) > 5]
            for desc in obj_descs:
                statements.append(StatementItem(
                    text=desc,
                    frame_id=frame_id,
                    video_id=video_id,
                    scene_id=scene_id
                ))

        # Bước 2.5: Làm giàu object bằng cách phát hiện và thêm các nhóm đối tượng mới (enriched objects)
        # 🧠 Prompt: phát hiện các nhóm đối tượng có liên hệ, tương đồng, hoặc tạo thành một thực thể lớn hơn; thêm tên nhóm vào objects để các bước sau cũng xử lý nhóm này
        if len(objects) >= 2:
            prompt_group = (
                f"Có các đối tượng sau trong ảnh: {', '.join(objects)}. "
                f"Hãy liệt kê các nhóm đối tượng có thể được xem là một thực thể hoặc nhóm mới dựa trên sự tương đồng, liên hệ, hoặc tạo thành một khối/thực thể lớn hơn. "
                f"Mỗi nhóm trả về một dòng mô tả ngắn gọn về nhóm đó và các thành phần của nó. Nếu không có nhóm nào thì trả về rỗng."
            )
            group_text = self._run_llava_inference(frame, prompt_group).strip()
            group_descs = re.split(r'(?<=[.!?])\s+', group_text)
            group_descs = [desc for desc in group_descs if len(desc.strip()) > 5]
            # Thêm tên nhóm vào objects để xử lý tiếp các bước sau (ví dụ: quan hệ giữa nhóm và object khác)
            enriched_objects = []
            for desc in group_descs:
                # 📝 Tìm tên nhóm (giả định tên nhóm là cụm đầu tiên trước dấu hai chấm hoặc dấu phẩy)
                group_name = desc.split(':')[0].split(',')[0].strip() if ':' in desc or ',' in desc else desc[:30].strip()
                if group_name and group_name not in objects:
                    enriched_objects.append(group_name)
                statements.append(StatementItem(
                    text=desc,
                    frame_id=frame_id,
                    video_id=video_id,
                    scene_id=scene_id
                ))
            objects.extend(enriched_objects)  # ⚡ Thêm nhóm vào objects để các bước sau cũng xử lý các nhóm này

        # Bước 3: Sinh statement về quan hệ giữa các object (bao gồm cả object gốc và enriched object nếu có >=2 object)
        if len(objects) >= 2:
            # 📝 Prompt chi tiết: mô tả về loại quan hệ (không gian, hành động, tương tác, vai trò, cảm xúc, v.v.) giữa tất cả các object (bao gồm cả nhóm)
            prompt_rel = (
                f"Mô tả chi tiết các mối quan hệ hoặc tương tác giữa các đối tượng sau trong ảnh: {', '.join(objects)}. "
                f"Bao gồm các khía cạnh như: vị trí tương đối, hành động tương tác, vai trò, cảm xúc, trạng thái, hoặc bất kỳ hình thức liên kết nào khác. "
                f"Trả về mỗi quan hệ là một câu riêng biệt, không giải thích thêm."
            )
            rel_text = self._run_llava_inference(frame, prompt_rel).strip()
            rel_statements = re.split(r'(?<=[.!?])\s+', rel_text)
            rel_statements = [s for s in rel_statements if len(s.strip()) > 5]
            for rel_stmt in rel_statements:
                statements.append(StatementItem(
                    text=rel_stmt,
                    frame_id=frame_id,
                    video_id=video_id,
                    scene_id=scene_id
                ))

        return statements
