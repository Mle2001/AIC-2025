from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: Optional[bool] = False
    model: Optional[str] = "grok-1"
    temperature: Optional[float] = 0.7
    openai_api_key: Optional[str] = None  # Thêm trường cho OpenAI API key (nếu cần truyền động)

class ChatResponse(BaseModel):
    response: str
    videos: List[dict] = []
    session_id: str
    model: Optional[str] = None
    usage: Optional[dict] = None
