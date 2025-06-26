from fastapi import APIRouter, WebSocket, Depends, HTTPException
from fastapi.responses import StreamingResponse
from api.models.chat_models import ChatRequest, ChatResponse
from api.services.chat_service import ChatService
from api.middleware.auth import get_current_user
import asyncio
import json

router = APIRouter()
chat_service = ChatService()

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, user=Depends(get_current_user)):
    """
    Xử lý chat message, trả về AI response và video results.
    """
    response = await chat_service.process_message(request, user)
    return ChatResponse(**response)

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    # Streaming response từ chat_service
    async for chunk in chat_service.stream_response(ChatRequest(message="", session_id=session_id), user_id="ws-user"):
        await websocket.send_text(chunk)
    await websocket.close()

@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str, limit: int = 50):
    # Dummy history, thực tế lấy từ DB hoặc agent
    return {"history": [], "session_id": session_id}
