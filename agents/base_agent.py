# /agents/base_agent.py
"""
Module chứa lớp cơ sở trừu tượng cho tất cả các agent trong hệ thống.
Phiên bản này được nâng cấp để tích hợp với framework Agno và bổ sung
các tiện ích về theo dõi hiệu năng và quản lý trạng thái.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from agno.agent import Agent
from agno.tools import BaseTool

logger = logging.getLogger(__name__)

class BaseAgent(Agent, ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các agent trong dự án.

    Kế thừa từ `agno.agent.Agent` để tận dụng khả năng tương tác với LLM và tools,
    đồng thời bổ sung các phương thức chung để đảm bảo tính nhất quán, khả năng
    ghi log, theo dõi hiệu năng và xử lý lỗi.
    """

    def __init__(
        self,
        name: str,
        role: str,
        instructions: List[str],
        model: Optional[Any] = None,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        """
        Khởi tạo một agent cơ sở.

        Args:
            name (str): Tên định danh duy nhất cho agent.
            role (str): Mô tả ngắn gọn về vai trò, chức năng của agent.
            instructions (List[str]): Danh sách các chỉ dẫn cho agent.
            model (Optional[Any]): Mô hình LLM mà agent sẽ sử dụng (ví dụ: OpenAIChat).
            tools (Optional[List[BaseTool]]): Danh sách các công cụ mà agent có thể sử dụng.
            config (Optional[Dict[str, Any]]): Cấu hình tùy chỉnh cho agent.
            **kwargs: Các tham số khác để truyền vào lớp `agno.agent.Agent`.
        """
        # Kết hợp vai trò vào trong chỉ dẫn chung cho Agno Agent
        full_instructions = [f"Vai trò của bạn là: {role}"] + (instructions or [])

        # Gọi hàm khởi tạo của lớp cha (Agno Agent)
        super().__init__(
            name=name,
            model=model,
            instructions=full_instructions,
            tools=tools or [],
            **kwargs
        )
        
        # Giữ lại các thuộc tính hữu ích từ phiên bản của bạn
        self.role = role
        self.config = config or {}
        self.performance_metrics: List[Dict[str, Any]] = []
        
        # Sử dụng logger có sẵn từ agno.Agent, không cần tạo mới
        self.logger.info(f"Agent '{self.name}' (Role: {self.role}) đã được khởi tạo.")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Phương thức xử lý chính của agent.

        Đây là một phương thức trừu tượng và phải được triển khai bởi tất cả các lớp con.
        Nó định nghĩa logic cốt lõi của agent.

        Args:
            input_data (Dict[str, Any]): Dữ liệu đầu vào cho agent.
            **kwargs: Các tham số tùy chọn khác.

        Returns:
            Dict[str, Any]: Kết quả xử lý của agent.
        """
        pass

    def log_performance(self, operation: str, duration: float, status: str):
        """
        Ghi lại một số liệu hiệu năng cho một hoạt động.

        Args:
            operation (str): Tên của hoạt động (ví dụ: 'call_llm', 'run_tool').
            duration (float): Thời gian thực thi (tính bằng giây).
            status (str): Trạng thái của hoạt động ('success' hoặc 'failed').
        """
        metric = {
            "agent": self.name,
            "operation": operation,
            "duration": round(duration, 4),
            "status": status,
            "timestamp": time.time()
        }
        self.performance_metrics.append(metric)
        self.logger.info(f"PERF_METRIC | Agent: {self.name} | Operation: {operation} | Status: {status} | Duration: {duration:.4f}s")

    @contextmanager
    def performance_tracker(self, operation: str):
        """
        Một context manager để dễ dàng theo dõi hiệu năng của một khối mã.

        Sử dụng:
            with self.performance_tracker("my_operation"):
                # code cần đo thời gian
                time.sleep(1)

        Args:
            operation (str): Tên của hoạt động đang được đo.
        """
        start_time = time.perf_counter()
        try:
            yield
            end_time = time.perf_counter()
            self.log_performance(operation, end_time - start_time, "success")
        except Exception:
            end_time = time.perf_counter()
            self.log_performance(operation, end_time - start_time, "failed")
            # Ném lại ngoại lệ sau khi đã ghi log
            raise

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý lỗi một cách nhất quán, ghi log và trả về một cấu trúc lỗi chuẩn.

        Args:
            error (Exception): Đối tượng ngoại lệ đã xảy ra.
            context (Dict[str, Any]): Ngữ cảnh tại thời điểm xảy ra lỗi.

        Returns:
            Dict[str, Any]: Một dictionary chứa thông tin về lỗi.
        """
        self.logger.error(f"ERROR | Agent: {self.name} | Error: {str(error)} | Context: {context}", exc_info=True)
        return {
            "status": "error",
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "context": context
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Trả về trạng thái và thông tin hiện tại của agent.
        Hữu ích cho việc giám sát (monitoring).

        Returns:
            Dict[str, Any]: Một dictionary chứa thông tin trạng thái.
        """
        return {
            "name": self.name,
            "role": self.role,
            "tools_available": [tool.name for tool in self.tools],
            "performance_summary": self.get_performance_summary()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Tổng hợp và trả về các số liệu hiệu năng đã ghi nhận.

        Returns:
            Dict[str, Any]: Thống kê về hiệu năng của agent.
        """
        if not self.performance_metrics:
            return {"total_operations": 0, "message": "No performance metrics recorded yet."}
        
        total_ops = len(self.performance_metrics)
        successful_ops = sum(1 for m in self.performance_metrics if m['status'] == 'success')
        failed_ops = total_ops - successful_ops
        total_duration = sum(m['duration'] for m in self.performance_metrics)
        avg_duration = total_duration / total_ops if total_ops > 0 else 0

        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "total_duration_seconds": round(total_duration, 4),
            "average_duration_seconds": round(avg_duration, 4)
        }
