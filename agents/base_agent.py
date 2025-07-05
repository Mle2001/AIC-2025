"""
Base Agent class cho tất cả agents trong hệ thống AI Challenge 2.0
Cung cấp common functionality và interface cho preprocessing và conversational agents
"""

from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage
import logging
import time

# Structured output models cho base responses
class AgentResponse(BaseModel):
    """Base response model cho tất cả agents"""
    agent_name: str = Field(..., description="Tên của agent")
    task_type: str = Field(..., description="Loại task được thực hiện")
    status: str = Field(..., description="Trạng thái: success, error, processing")
    result: Dict[str, Any] = Field(default_factory=dict, description="Kết quả chi tiết")
    execution_time: float = Field(..., description="Thời gian thực hiện (seconds)")
    error_message: Optional[str] = Field(None, description="Thông báo lỗi nếu có")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung")

class BaseAIAgent(ABC):
    """
    Base class cho tất cả AI agents trong hệ thống
    Cung cấp common functionality và interface chuẩn
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model_type: str = "gpt-4o",
        enable_memory: bool = True,
        enable_storage: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        self.name = name
        self.description = description
        self.model_type = model_type
        self.session_id = session_id or f"{name}_{int(time.time())}"
        self.user_id = user_id or "system"
        
        # Setup model
        self.model = self._setup_model(model_type)
        
        # Setup memory and storage
        self.memory = self._setup_memory() if enable_memory else None
        self.storage = self._setup_storage() if enable_storage else None
        
        # Initialize Agno agent
        self.agent = self._create_agent(**kwargs)
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{name}")
        
    def _setup_model(self, model_type: str):
        """Setup AI model dựa trên type"""
        if model_type.startswith("gpt"):
            return OpenAIChat(id=model_type)
        elif model_type.startswith("claude"):
            return Claude(id=model_type)
        else:
            # Default to GPT-4o
            return OpenAIChat(id="gpt-4o")
    
    def _setup_memory(self) -> Memory:
        """Setup memory cho agent"""
        return Memory(
            model=OpenAIChat(id="gpt-4o-mini"),  # Cheaper model cho memory tasks
            db=None  # Will use default in-memory storage
        )
    
    def _setup_storage(self) -> SqliteStorage:
        """Setup storage cho agent sessions"""
        return SqliteStorage(
            table_name=f"agent_sessions_{self.name.lower()}",
            db_file=f"tmp/{self.name.lower()}_storage.db"
        )
    
    def _create_agent(self, **kwargs) -> Agent:
        """Tạo Agno agent với config cơ bản"""
        return Agent(
            name=self.name,
            model=self.model,
            description=self.description,
            memory=self.memory,
            storage=self.storage,
            session_id=self.session_id,
            # Common settings
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            enable_agentic_memory=True if self.memory else False,
            add_history_to_messages=True,
            num_history_runs=3,
            **kwargs
        )
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> AgentResponse:
        """
        Main processing method - must be implemented by subclasses
        """
        pass
    
    def _create_response(
        self,
        task_type: str,
        status: str,
        result: Dict[str, Any],
        execution_time: float,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Helper để tạo standardized response"""
        return AgentResponse(
            agent_name=self.name,
            task_type=task_type,
            status=status,
            result=result,
            execution_time=execution_time,
            error_message=error_message,
            metadata=metadata or {}
        )
    
    def run_with_timing(self, message: str, **kwargs) -> AgentResponse:
        """Run agent với timing và error handling"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting task: {message[:100]}...")
            
            # Run the agent
            response = self.agent.run(message, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Extract result from agent response
            result = {
                "content": response.content if hasattr(response, 'content') else str(response),
                "run_id": response.run_id if hasattr(response, 'run_id') else None,
                "messages": len(self.agent.get_messages_for_session()) if self.agent else 0
            }
            
            self.logger.info(f"Task completed in {execution_time:.2f}s")
            
            return self._create_response(
                task_type="agent_run",
                status="success",
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Task failed after {execution_time:.2f}s: {error_msg}")
            
            return self._create_response(
                task_type="agent_run",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=error_msg
            )
    
    def get_session_info(self) -> Dict[str, Any]:
        """Lấy thông tin session hiện tại"""
        return {
            "agent_name": self.name,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "model_type": self.model_type,
            "has_memory": self.memory is not None,
            "has_storage": self.storage is not None,
            "message_count": len(self.agent.get_messages_for_session()) if self.agent else 0
        }
    
    def clear_session(self):
        """Clear session data"""
        if self.agent:
            # Clear messages
            self.agent.session_state = {}
        self.logger.info("Session cleared")
    
    def set_instructions(self, instructions: Union[str, List[str]]):
        """Update agent instructions"""
        if self.agent:
            self.agent.instructions = instructions
            self.logger.info("Instructions updated")
    
    def add_tools(self, tools: List[Any]):
        """Add tools to agent"""
        if self.agent:
            current_tools = getattr(self.agent, 'tools', [])
            self.agent.tools = current_tools + tools
            self.logger.info(f"Added {len(tools)} tools to agent")

class PreprocessingAgent(BaseAIAgent):
    """
    Base class cho preprocessing agents
    Specialized cho video processing tasks
    """
    
    def __init__(self, name: str, **kwargs):
        # Default settings cho preprocessing
        kwargs.setdefault('model_type', 'gpt-4o')  # Mạnh hơn cho preprocessing
        kwargs.setdefault('enable_memory', False)  # Không cần memory cho batch processing
        kwargs.setdefault('enable_storage', True)  # Cần track processing state
        
        super().__init__(name=name, description=f"Preprocessing agent: {name}", **kwargs)
    
    def process_batch(self, batch_data: List[Any], **kwargs) -> List[AgentResponse]:
        """Process batch data - common cho preprocessing tasks"""
        results = []
        
        for i, item in enumerate(batch_data):
            self.logger.info(f"Processing item {i+1}/{len(batch_data)}")
            result = self.process(item, **kwargs)
            results.append(result)
            
            # Optional: Add delay giữa items để tránh rate limiting
            if kwargs.get('delay_between_items', 0) > 0:
                time.sleep(kwargs['delay_between_items'])
        
        return results

class ConversationalAgent(BaseAIAgent):
    """
    Base class cho conversational agents  
    Specialized cho real-time chat tasks
    """
    
    def __init__(self, name: str, **kwargs):
        # Default settings cho conversational
        kwargs.setdefault('model_type', 'gpt-4o')  
        kwargs.setdefault('enable_memory', True)  # Cần memory cho conversation
        kwargs.setdefault('enable_storage', True)  # Cần persist conversations
        
        super().__init__(name=name, description=f"Conversational agent: {name}", **kwargs)
        
        # Enhanced settings cho conversation
        if self.agent:
            self.agent.add_history_to_messages = True
            self.agent.num_history_runs = 5  # More history cho better context
            self.agent.enable_agentic_memory = True
    
    def chat(self, message: str, **kwargs) -> AgentResponse:
        """Specialized chat method"""
        return self.run_with_timing(message, **kwargs)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history cho debugging"""
        if not self.agent:
            return []
        
        messages = self.agent.get_messages_for_session()
        return [
            {
                "role": msg.role,
                "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            }
            for msg in messages
        ]