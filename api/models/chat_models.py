# api/models/chat_models.py
"""
Chat Models - Pydantic models cho chat system
Dev2: API Data Models - định nghĩa structure cho chat requests/responses
Current: 2025-07-03 14:14:50 UTC, User: xthanh1910
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

#==========================================================================================================================================
# ENUMS CHO CHAT SYSTEM
#==========================================================================================================================================

class MessageType(str, Enum):
    """
    Loại message trong chat system
    """
    USER = "user"              # Message từ user
    ASSISTANT = "assistant"    # Response từ AI assistant
    SYSTEM = "system"         # System messages (notifications, etc.)
    ERROR = "error"           # Error messages
    QUICK = "quick"           # Quick chat messages

class ChatMode(str, Enum):
    """
    Chế độ chat khác nhau
    """
    REGULAR = "regular"        # Chat thường với full context
    QUICK = "quick"           # Quick chat không cần context phức tạp
    DEEP = "deep"             # Deep exploration mode
    EDUCATIONAL = "educational" # Educational mode với explanations
    RESEARCH = "research"      # Research mode với detailed analysis

class SessionStatus(str, Enum):
    """
    Trạng thái của chat session
    """
    ACTIVE = "active"         # Session đang hoạt động
    INACTIVE = "inactive"     # Session tạm dừng
    EXPIRED = "expired"       # Session đã hết hạn
    CLOSED = "closed"         # Session đã đóng

# ================================
# REQUEST MODELS (từ Frontend Dev4)
# ================================

class ChatRequest(BaseModel):
    """
    Request model cho chat message
    Frontend Dev4 sẽ gửi model này
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Nội dung tin nhắn từ user"
    )
    session_id: Optional[str] = Field(
        None,
        description="ID của session chat (auto-generate nếu không có)"
    )
    user_id: Optional[str] = Field(
        None,
        description="ID của user (sẽ được lấy từ authentication)"
    )
    chat_mode: ChatMode = Field(
        ChatMode.REGULAR,
        description="Chế độ chat"
    )
    context_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Context data bổ sung cho message"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        {},
        description="Metadata bổ sung (IP, user-agent, etc.)"
    )

    @validator('message')
    def validate_message(cls, v):
        """
        Validate message content
        """
        if not v or not v.strip():
            raise ValueError('Message không được để trống')

        # Remove excessive whitespace
        v = ' '.join(v.split())

        # Check for potentially harmful content (basic)
        harmful_patterns = ['<script', 'javascript:', 'data:text/html']
        v_lower = v.lower()
        for pattern in harmful_patterns:
            if pattern in v_lower:
                raise ValueError('Message chứa nội dung không được phép')

        return v

    class Config:
        schema_extra = {
            "example": {
                "message": "Tôi muốn tìm video về cách nấu phở",
                "session_id": "session_123",
                "chat_mode": "regular",
                "context_data": {
                    "previous_topic": "cooking",
                    "user_preferences": ["vietnamese_food"]
                },
                "metadata": {
                    "source": "web_app",
                    "language": "vi"
                }
            }
        }

class QuickChatRequest(BaseModel):
    """
    Request model cho quick chat (response nhanh)
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Câu hỏi ngắn cho quick response"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID (optional cho quick chat)"
    )

    @validator('message')
    def validate_quick_message(cls, v):
        """
        Validate quick message (stricter limits)
        """
        v = v.strip()
        if len(v) < 3:
            raise ValueError('Quick message phải có ít nhất 3 ký tự')
        return v

    class Config:
        schema_extra = {
            "example": {
                "message": "Video nào hay về nấu ăn?",
                "session_id": "quick_session_456"
            }
        }

class WebSocketChatMessage(BaseModel):
    """
    Model cho WebSocket chat messages
    """
    type: str = Field(
        ...,
        description="Loại message: 'user_message', 'get_history', 'ping'"
    )
    message: Optional[str] = Field(
        None,
        description="Nội dung message (cho type='user_message')"
    )
    session_id: str = Field(
        ...,
        description="Session ID cho WebSocket connection"
    )
    user_id: str = Field(
        ...,
        description="User ID"
    )
    data: Optional[Dict[str, Any]] = Field(
        {},
        description="Additional data"
    )

    class Config:
        schema_extra = {
            "example": {
                "type": "user_message",
                "message": "Tìm video về du lịch Đà Nẵng",
                "session_id": "ws_session_789",
                "user_id": "user_123",
                "data": {"timestamp": "2025-07-03T14:14:50Z"}
            }
        }

# ================================
# RESPONSE MODELS (cho Frontend Dev4)
# ================================

class MediaReference(BaseModel):
    """
    Reference tới media content trong response
    """
    media_id: str = Field(..., description="ID của media")
    media_type: str = Field(..., description="Loại media: video, image, audio")
    title: str = Field(..., description="Tiêu đề media")
    thumbnail_url: Optional[str] = Field(None, description="URL thumbnail")
    duration: Optional[int] = Field(None, description="Thời lượng (seconds)")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Điểm relevance (0-1)"
    )
    start_time: Optional[int] = Field(None, description="Thời điểm bắt đầu relevant")
    end_time: Optional[int] = Field(None, description="Thời điểm kết thúc relevant")

    class Config:
        schema_extra = {
            "example": {
                "media_id": "video_123",
                "media_type": "video",
                "title": "Cách nấu phở bò truyền thống",
                "thumbnail_url": "https://example.com/thumb.jpg",
                "duration": 600,
                "relevance_score": 0.95,
                "start_time": 120,
                "end_time": 180
            }
        }

class AgentInfo(BaseModel):
    """
    Thông tin về agents đã sử dụng trong processing
    """
    agents_used: int = Field(..., description="Số lượng agents đã sử dụng")
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tỷ lệ thành công của agents"
    )
    flow_status: str = Field(..., description="Trạng thái của conversation flow")
    query_understanding: Optional[Dict[str, Any]] = Field(
        None,
        description="Kết quả query understanding"
    )
    video_results_count: int = Field(
        0,
        description="Số lượng video tìm được"
    )
    processing_stages: Optional[List[str]] = Field(
        None,
        description="Các stages đã thực hiện"
    )

    class Config:
        schema_extra = {
            "example": {
                "agents_used": 5,
                "success_rate": 0.92,
                "flow_status": "completed",
                "query_understanding": {
                    "intent": "search_videos",
                    "entities": ["phở", "nấu ăn", "việt nam"],
                    "confidence": 0.95
                },
                "video_results_count": 12,
                "processing_stages": [
                    "query_understanding",
                    "video_retrieval",
                    "content_explanation",
                    "response_synthesis"
                ]
            }
        }

class ConversationContext(BaseModel):
    """
    Context thông tin về conversation
    """
    turn_count: int = Field(..., description="Số lượt hội thoại")
    mentioned_videos: int = Field(..., description="Số video đã đề cập")
    active_entities: int = Field(..., description="Số entities đang active")
    session_duration_minutes: Optional[float] = Field(
        None,
        description="Thời lượng session (phút)"
    )
    last_topic: Optional[str] = Field(None, description="Topic cuối cùng")

    class Config:
        schema_extra = {
            "example": {
                "turn_count": 5,
                "mentioned_videos": 3,
                "active_entities": 7,
                "session_duration_minutes": 12.5,
                "last_topic": "cooking_vietnamese_food"
            }
        }

class ChatResponse(BaseModel):
    """
    Response chính cho chat request
    Dev2 trả về model này cho Frontend Dev4
    """
    success: bool = Field(..., description="Request có thành công không")
    response: str = Field(..., description="Câu trả lời từ AI assistant")
    session_id: str = Field(..., description="Session ID")
    processing_time: float = Field(..., description="Thời gian xử lý (seconds)")

    # Optional advanced data
    agent_info: Optional[AgentInfo] = Field(
        None,
        description="Thông tin về agents processing"
    )
    media_references: List[MediaReference] = Field(
        [],
        description="Danh sách media được reference"
    )
    follow_up_suggestions: List[str] = Field(
        [],
        description="Gợi ý câu hỏi tiếp theo"
    )
    conversation_context: Optional[ConversationContext] = Field(
        None,
        description="Context của conversation"
    )

    # Error handling
    error_code: Optional[str] = Field(None, description="Mã lỗi nếu có")
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Chi tiết lỗi"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "response": "Tôi tìm thấy 5 video về cách nấu phở bò. Video phổ biến nhất là 'Cách nấu phở bò truyền thống' với 1.2M views. Bạn muốn xem video nào cụ thể?",
                "session_id": "session_123",
                "processing_time": 2.45,
                "agent_info": {
                    "agents_used": 4,
                    "success_rate": 0.95,
                    "flow_status": "completed",
                    "video_results_count": 5
                },
                "media_references": [
                    {
                        "media_id": "video_456",
                        "media_type": "video",
                        "title": "Cách nấu phở bò truyền thống",
                        "relevance_score": 0.98
                    }
                ],
                "follow_up_suggestions": [
                    "Cho tôi xem chi tiết video nấu phở",
                    "Tìm video về cách làm nước dũng phở",
                    "Video nào dạy cách thái thịt bò?"
                ],
                "conversation_context": {
                    "turn_count": 3,
                    "mentioned_videos": 5,
                    "active_entities": 4
                }
            }
        }

class QuickChatResponse(BaseModel):
    """
    Response cho quick chat
    """
    success: bool = Field(..., description="Request có thành công không")
    response: str = Field(..., description="Quick response")
    session_id: str = Field(..., description="Session ID")
    processing_time: float = Field(..., description="Thời gian xử lý")
    mode: str = Field(default="quick", description="Chat mode")
    media_references: List[MediaReference] = Field(
        [],
        description="Media references (if any)"
    )
    error_code: Optional[str] = Field(None, description="Error code nếu có")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "response": "Có 23 video hay về nấu ăn. Top video: 'Món ngon mỗi ngày' và 'Bếp nhà Việt'.",
                "session_id": "quick_session_456",
                "processing_time": 0.85,
                "mode": "quick",
                "media_references": []
            }
        }

class WebSocketChatResponse(BaseModel):
    """
    Response cho WebSocket chat
    """
    type: str = Field(..., description="Loại response")
    message: Optional[str] = Field(None, description="Message content")
    session_id: str = Field(..., description="Session ID")
    processing_time: Optional[float] = Field(None, description="Processing time")
    media_references: List[MediaReference] = Field([], description="Media refs")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    error: Optional[str] = Field(None, description="Error message nếu có")

    class Config:
        schema_extra = {
            "example": {
                "type": "response",
                "message": "Đây là 3 video du lịch Đà Nẵng hay nhất...",
                "session_id": "ws_session_789",
                "processing_time": 1.23,
                "media_references": [],
                "data": {"timestamp": "2025-07-03T14:14:50Z"}
            }
        }

# ================================
# CHAT HISTORY MODELS
# ================================

class ChatMessage(BaseModel):
    """
    Model cho một message trong chat history
    """
    message_id: Optional[str] = Field(None, description="ID của message")
    message_type: MessageType = Field(..., description="Loại message")
    content: str = Field(..., description="Nội dung message")
    timestamp: datetime = Field(..., description="Thời gian gửi message")
    processing_time: Optional[float] = Field(None, description="Thời gian xử lý")
    media_references: List[MediaReference] = Field([], description="Media refs")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")

    class Config:
        schema_extra = {
            "example": {
                "message_id": "msg_123",
                "message_type": "user",
                "content": "Tìm video về du lịch",
                "timestamp": "2025-07-03T14:14:50Z",
                "processing_time": None,
                "media_references": [],
                "metadata": {"source": "web"}
            }
        }

class ChatHistory(BaseModel):
    """
    Lịch sử chat của một session
    """
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Danh sách messages")
    total_messages: int = Field(..., description="Tổng số messages")
    session_created: datetime = Field(..., description="Thời gian tạo session")
    last_activity: datetime = Field(..., description="Hoạt động cuối")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "messages": [
                    {
                        "message_type": "user",
                        "content": "Xin chào",
                        "timestamp": "2025-07-03T14:14:50Z"
                    },
                    {
                        "message_type": "assistant",
                        "content": "Chào bạn! Tôi có thể giúp gì?",
                        "timestamp": "2025-07-03T14:14:52Z",
                        "processing_time": 1.2
                    }
                ],
                "total_messages": 2,
                "session_created": "2025-07-03T14:14:50Z",
                "last_activity": "2025-07-03T14:14:52Z"
            }
        }

# ================================
# SESSION MANAGEMENT MODELS
# ================================

class ChatSession(BaseModel):
    """
    Model cho chat session info
    """
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    status: SessionStatus = Field(..., description="Trạng thái session")
    created_at: datetime = Field(..., description="Thời gian tạo")
    last_activity: datetime = Field(..., description="Hoạt động cuối")
    message_count: int = Field(..., description="Số lượng messages")
    session_name: Optional[str] = Field(None, description="Tên session")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "user_id": "user_456",
                "status": "active",
                "created_at": "2025-07-03T14:14:50Z",
                "last_activity": "2025-07-03T14:20:30Z",
                "message_count": 8,
                "session_name": "Chat về nấu ăn"
            }
        }

class UserSessions(BaseModel):
    """
    Danh sách sessions của user
    """
    user_id: str = Field(..., description="User ID")
    sessions: List[ChatSession] = Field(..., description="Danh sách sessions")
    total_sessions: int = Field(..., description="Tổng số sessions")
    active_sessions: int = Field(..., description="Số sessions đang active")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_456",
                "sessions": [
                    {
                        "session_id": "session_123",
                        "status": "active",
                        "message_count": 8,
                        "session_name": "Chat về nấu ăn"
                    }
                ],
                "total_sessions": 5,
                "active_sessions": 2
            }
        }

# ================================
# ERROR MODELS
# ================================

class ChatError(BaseModel):
    """
    Model cho chat errors
    """
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Mã lỗi")
    error_message: str = Field(..., description="Thông báo lỗi")
    session_id: Optional[str] = Field(None, description="Session ID nếu có")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Thời gian lỗi")
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết lỗi")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "PROCESSING_TIMEOUT",
                "error_message": "Quá thời gian xử lý request",
                "session_id": "session_123",
                "timestamp": "2025-07-03T14:14:50Z",
                "details": {
                    "timeout_seconds": 30,
                    "stage": "video_retrieval"
                }
            }
        }

# ================================
# ANALYTICS MODELS
# ================================

class ChatAnalytics(BaseModel):
    """
    Analytics data cho chat system
    """
    total_sessions: int = Field(..., description="Tổng số sessions")
    total_messages: int = Field(..., description="Tổng số messages")
    active_sessions: int = Field(..., description="Sessions đang active")
    avg_messages_per_session: float = Field(..., description="Trung bình messages/session")
    avg_response_time: float = Field(..., description="Thời gian response trung bình")
    top_topics: List[str] = Field(..., description="Chủ đề phổ biến")
    user_satisfaction: Optional[float] = Field(None, description="Điểm hài lòng user")

    class Config:
        schema_extra = {
            "example": {
                "total_sessions": 1250,
                "total_messages": 8945,
                "active_sessions": 45,
                "avg_messages_per_session": 7.2,
                "avg_response_time": 2.1,
                "top_topics": ["nấu ăn", "du lịch", "giải trí"],
                "user_satisfaction": 4.3
            }
        }

# ================================
# UTILITY FUNCTIONS
# ================================

def create_success_response(
    response_text: str,
    session_id: str,
    processing_time: float,
    **kwargs
) -> ChatResponse:
    """
    Helper function tạo success response
    """
    return ChatResponse(
        success=True,
        response=response_text,
        session_id=session_id,
        processing_time=processing_time,
        **kwargs
    )

def create_error_response(
    error_message: str,
    error_code: str,
    session_id: Optional[str] = None,
    **kwargs
) -> ChatError:
    """
    Helper function tạo error response
    """
    return ChatError(
        error_code=error_code,
        error_message=error_message,
        session_id=session_id,
        **kwargs
    )

def create_quick_response(
    response_text: str,
    session_id: str,
    processing_time: float
) -> QuickChatResponse:
    """
    Helper function tạo quick chat response
    """
    return QuickChatResponse(
        success=True,
        response=response_text,
        session_id=session_id,
        processing_time=processing_time,
        mode="quick"
    )

# ================================
# EXPORTS
# ================================

__all__ = [
    # Enums
    "MessageType",
    "ChatMode",
    "SessionStatus",

    # Request Models
    "ChatRequest",
    "QuickChatRequest",
    "WebSocketChatMessage",

    # Response Models
    "ChatResponse",
    "QuickChatResponse",
    "WebSocketChatResponse",
    "MediaReference",
    "AgentInfo",
    "ConversationContext",

    # History Models
    "ChatMessage",
    "ChatHistory",

    # Session Models
    "ChatSession",
    "UserSessions",

    # Error Models
    "ChatError",

    # Analytics Models
    "ChatAnalytics",

    # Utility Functions
    "create_success_response",
    "create_error_response",
    "create_quick_response"
]