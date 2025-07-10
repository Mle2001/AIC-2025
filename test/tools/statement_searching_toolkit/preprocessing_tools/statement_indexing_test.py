# ============================
# 📦 IMPORTS - Giải thích mục đích từng gói
# ============================
import pytest  # 👉 Thư viện test cho Python
from tools.statement_searching_toolkit.preprocessing_tools.statement_indexing import StatementIndexingTools, StatementItem, EmbeddedStatement, FrameInfo
from PIL import Image  # 👉 Xử lý ảnh giả lập cho test

# ============================
# 🧪 TEST CHO StatementIndexingTools
# ============================
@pytest.fixture
def statement_indexing_tools():
    # Khởi tạo tool với model mặc định
    return StatementIndexingTools()

@pytest.fixture
def dummy_frame_info():
    # Tạo frame từ file ảnh thật (ví dụ: test_image.jpg trong thư mục test/assets)
    img_path = "d:/AIC-2025/AIC-2025/test/assets/sample.png"  # Đường dẫn ảnh mẫu
    img = Image.open(img_path)
    return FrameInfo(frame=img, frame_id="f1", video_id="v1", scene_id="s1")

@pytest.fixture
def dummy_statements():
    # Tạo danh sách StatementItem giả lập
    return [
        StatementItem(text="A cat is sitting on the sofa.", frame_id="f1", video_id="v1", scene_id="s1"),
        StatementItem(text="A dog is lying next to the cat.", frame_id="f1", video_id="v1", scene_id="s1")
    ]

# ============================
# 🧪 Test extract_statements_from_frame
# ============================
@pytest.mark.asyncio
async def test_extract_statements_from_frame(statement_indexing_tools, dummy_frame_info):
    
    statements = await statement_indexing_tools.extract_statements_from_frame(dummy_frame_info)
    assert isinstance(statements, list)
    assert all(isinstance(s, StatementItem) for s in statements)
    # Có thể kiểm tra thêm số lượng hoặc nội dung nếu cần

# ============================
# 🧪 Test embed_statements
# ============================
def test_embed_statements(statement_indexing_tools, dummy_statements):
    embedded = statement_indexing_tools.embed_statements(dummy_statements)
    assert isinstance(embedded, list)
    assert all(isinstance(e, EmbeddedStatement) for e in embedded)
    assert all(hasattr(e, 'embedding') and isinstance(e.embedding, list) for e in embedded)

# ============================
# 🧪 Test save_embeddings_to_vectordb
# ============================
def test_save_embeddings_to_vectordb(statement_indexing_tools, dummy_statements):
    embedded = statement_indexing_tools.embed_statements(dummy_statements)
    result = statement_indexing_tools.save_embeddings_to_vectordb(embedded, collection_name="test_statements")
    assert result is True
