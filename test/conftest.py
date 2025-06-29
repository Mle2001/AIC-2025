# /test/conftest.py
"""
Tệp cấu hình chung cho các bài kiểm thử Pytest.
Tệp này sẽ được Pytest tự động phát hiện và sử dụng.
"""

import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def passthrough_agno_tool():
    """
    Fixture này sẽ tự động chạy cho TẤT CẢ các bài kiểm thử.

    Nó thực hiện một việc rất quan trọng: "vá" (patch) decorator `@tool` của agno.
    Thay vì để `@tool` biến các phương thức thành đối tượng đặc biệt, chúng ta
    thay thế nó bằng một decorator "rỗng" chỉ trả về hàm gốc.

    Kết quả: Trong môi trường kiểm thử, tất cả các phương thức như `detect_scenes`,
    `search_vectors`, v.v., sẽ trở thành các phương thức Python thông thường và
    có thể được gọi trực tiếp, giải quyết lỗi `TypeError: 'Function' object is not callable`.
    """
    # Đây là một decorator giả, nó nhận vào bất kỳ tham số nào của @tool
    # và trả về một decorator khác chỉ có nhiệm vụ trả về hàm gốc.
    def identity_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    # Sử dụng patch của unittest.mock để thay thế 'agno.tools.tool'
    # bằng decorator giả của chúng ta trong suốt quá trình chạy test.
    with patch('agno.tools.tool', new=identity_decorator):
        yield
