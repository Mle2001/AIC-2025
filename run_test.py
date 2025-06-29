# /run_tests.py
"""
Kịch bản chạy kiểm thử tùy chỉnh để giải quyết các vấn đề về môi trường.
Thay vì chạy 'pytest', bạn sẽ chạy 'python run_tests.py'.
"""

import pytest
from unittest.mock import patch
import sys

def main():
    """
    Hàm chính để cấu hình môi trường và chạy pytest.
    """
    print("🚀 applying agno @tool patch and running tests...")

    # Đây là một decorator giả, nó nhận vào bất kỳ tham số nào của @tool
    # và trả về một decorator khác chỉ có nhiệm vụ trả về hàm gốc.
    def identity_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    # Sử dụng patch của unittest.mock để thay thế 'agno.tools.tool'
    # bằng decorator giả của chúng ta.
    # 'patch' sẽ tự động được áp dụng trong khối 'with'.
    with patch('agno.tools.tool', new=identity_decorator):
        # Lấy các tham số dòng lệnh (nếu có), trừ tên của chính script này
        args = sys.argv[1:]
        
        # Nếu không có tham số nào được truyền, thêm '-v' để có output chi tiết
        if not args:
            args.append('-v')
            
        # Gọi hàm main của pytest với các tham số đã được xử lý
        exit_code = pytest.main(args)
        
        # Trả về exit code của pytest
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
