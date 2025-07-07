# /tools/retrieval/base_retriever.py
"""
Module chứa lớp cơ sở trừu tượng cho tất cả các công cụ truy xuất.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các retriever.

    Định nghĩa một interface chung cho việc kết nối và đóng tài nguyên,
    đảm bảo các retriever con tuân thủ một cấu trúc nhất quán.
    """
    def __init__(self):
        """
        Hàm khởi tạo sẽ gọi phương thức _connect để thiết lập kết nối.
        """
        self._connect()

    @abstractmethod
    def _connect(self):
        """
        Phương thức trừu tượng để thiết lập kết nối tới nguồn dữ liệu (DB, API, etc.).
        Các lớp con phải triển khai phương thức này.
        """
        pass

    def close(self):
        """
        Phương thức (không bắt buộc) để đóng các kết nối và giải phóng tài nguyên.
        Các lớp con có thể ghi đè phương thức này nếu cần.
        """
        logger.info(f"Retriever '{self.__class__.__name__}' không yêu cầu hành động đóng kết nối cụ thể.")
        pass

