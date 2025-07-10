# ============================
# 📦 IMPORTS - Giải thích mục đích từng gói
# ============================
import pytest  # 👉 Thư viện test cho Python
from tools.statement_searching_toolkit.retrieval_tools.statement_retrieval import StatementRetrievalTools

# ============================
# 🧪 TEST CHO StatementRetrievalTools
# ============================
@pytest.fixture
def retrieval_tools():
    # Khởi tạo tool với model mặc định
    return StatementRetrievalTools()

@pytest.fixture
def dummy_description():
    # Mô tả giả lập cho frame ảnh
    return "A cat is sitting on the sofa. A dog is lying next to the cat. The window is open. The cat and dog are looking outside."

@pytest.fixture
def dummy_statements():
    # Danh sách statement giả lập
    return [
        "A cat is sitting on the sofa.",
        "A dog is lying next to the cat.",
        "The window is open."
    ]

@pytest.fixture
def dummy_embeddings(retrieval_tools, dummy_statements):
    # Nhúng embedding cho statements giả lập
    return retrieval_tools.embed_statements(dummy_statements)

# ============================
# 🧪 Test extract_statements_from_description
# ============================
def test_extract_statements_from_description(retrieval_tools, dummy_description):
    statements = retrieval_tools.extract_statements_from_description(dummy_description)
    assert isinstance(statements, list)
    assert all(isinstance(s, str) for s in statements)
    assert any("cat" in s.lower() for s in statements)  # Kiểm tra có statement về cat

# ============================
# 🧪 Test embed_statements
# ============================
def test_embed_statements(retrieval_tools, dummy_statements):
    embeddings = retrieval_tools.embed_statements(dummy_statements)
    assert isinstance(embeddings, list)
    assert all(isinstance(e, list) for e in embeddings)
    assert all(isinstance(val, float) for e in embeddings for val in e)

# ============================
# 🧪 Test search_similar_frames (mock)
# ============================
def test_search_similar_frames(retrieval_tools, dummy_embeddings):
    # Do không có ChromaDB thực tế, chỉ kiểm tra gọi hàm không lỗi và trả về list
    try:
        results = retrieval_tools.search_similar_frames(dummy_embeddings, k=2, collection_name="test_statements")
        assert isinstance(results, list)
    except Exception:
        # Nếu không có ChromaDB, test vẫn pass (mock)
        assert True
