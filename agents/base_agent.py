import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    def __init__(self, model: Any, name: str, role: str, config: Dict[str, Any] = None):
        self.model = model
        self.name = name
        self.role = role
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.performance_metrics = []

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method phải được implement bởi subclasses.
        Xử lý input và trả về kết quả.
        """
        pass

    def log_performance(self, operation: str, duration: float, status: str):
        metric = {
            "agent": self.name,
            "operation": operation,
            "duration": duration,
            "status": status
        }
        self.performance_metrics.append(metric)
        self.logger.info(f"[{self.name}] {operation} - {status} - {duration:.2f}s")

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.error(f"Error in {self.name}: {str(error)} | Context: {context}")
        return {
            "status": "error",
            "error": str(error),
            "context": context
        }
