# api/routers/chat.py
"""
Chat Router - API endpoints cho conversation system
Dev2: API Integration & Routing
Current: 2025-07-03 11:48:36 UTC, User: xthanh1910
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from typing import Optional, List
import json
import time

# Import agents từ Dev1
from ..agents_manager import agents_manager
from agents.conversational.context_manager_agent import SessionContext

# Import models cho API (Dev2 tạo đơn giản)
from ..models.chat_models import ChatRequest, ChatResponse
from ..models.user_models import User

# Import services để connect với database Dev3
from ..services.chat_service import ChatService
from ..middleware.auth import get_current_user

# Initialize services
chat_service = ChatService()
router = APIRouter(prefix="/chat", tags=["chat"])

#==========================================================================================================================================
# MAIN CHAT ENDPOINTS
#==========================================================================================================================================
@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Main chat endpoint - gửi message và nhận response
    Tích hợp ConversationOrchestrator từ Dev1
    """
    try:
        # Lấy conversation orchestrator từ Dev1
        orchestrator = agents_manager.get_conversation_orchestrator()

        # Tạo session context
        session_context = SessionContext(
            session_id=request.session_id or f"session_{current_user.user_id}_{int(time.time())}",
            user_id=current_user.user_id,
            start_time=str(time.time()),
            conversation_turns=[],
            active_entities=[],
            mentioned_videos=[],
            search_history=[]
        )

        # Gọi orchestrator để xử lý conversation (Dev1's logic)
        result = orchestrator.process_conversation(
            user_query=request.message,
            session_context=session_context,
            user_id=current_user.user_id
        )

        # Check kết quả từ agents
        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        # Extract response data
        flow_data = result.result
        final_response = flow_data.get("final_response", {})
        main_answer = final_response.get("main_answer", "Xin lỗi, tôi gặp vấn đề khi xử lý câu hỏi của bạn.")

        # Lưu vào database thông qua Dev3's service
        await chat_service.save_chat_message(
            session_id=session_context.session_id,
            user_id=current_user.user_id,
            user_message=request.message,
            ai_response=main_answer,
            processing_time=result.execution_time
        )

        # Return API response cho Dev4's frontend
        return ChatResponse(
            response=main_answer,
            session_id=session_context.session_id,
            processing_time=result.execution_time,
            agent_info={
                "agents_used": len(flow_data.get("agent_executions", [])),
                "success_rate": flow_data.get("success_rate", 0),
                "flow_status": flow_data.get("status", "unknown")
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.post("/quick")
async def quick_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Quick chat endpoint - fast response mode
    """
    try:
        # Sử dụng quick_chat method từ orchestrator
        orchestrator = agents_manager.get_conversation_orchestrator()

        result = orchestrator.quick_chat(
            user_query=request.message,
            session_id=request.session_id or f"quick_{current_user.user_id}",
            user_id=current_user.user_id
        )

        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        flow_data = result.result
        final_response = flow_data.get("final_response", {})

        return {
            "response": final_response.get("main_answer", "No response available"),
            "session_id": request.session_id,
            "processing_time": result.execution_time,
            "mode": "quick"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick chat failed: {str(e)}")

#==========================================================================================================================================
# WEBSOCKET ENDPOINT
#==========================================================================================================================================

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint cho real-time chat
    """
    await websocket.accept()
    orchestrator = agents_manager.get_conversation_orchestrator()

    try:
        while True:
            # Nhận message từ frontend Dev4
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            user_id = message_data.get("user_id", "anonymous")

            # Xử lý với orchestrator
            session_context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                start_time=str(time.time()),
                conversation_turns=[],
                active_entities=[],
                mentioned_videos=[],
                search_history=[]
            )

            result = orchestrator.quick_chat(
                user_query=user_message,
                session_id=session_id,
                user_id=user_id
            )

            # Gửi response về frontend Dev4
            flow_data = result.result
            final_response = flow_data.get("final_response", {})

            await websocket.send_text(json.dumps({
                "type": "response",
                "message": final_response.get("main_answer", ""),
                "session_id": session_id,
                "processing_time": result.execution_time,
                "media_references": final_response.get("media_references", [])
            }))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error: {str(e)}"
        }))

#==========================================================================================================================================
# CHAT HISTORY ENDPOINTS
#==========================================================================================================================================

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Lấy lịch sử chat - connect với database Dev3
    """
    try:
        history = await chat_service.get_chat_history(session_id, current_user.user_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.delete("/history/{session_id}")
async def delete_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Xóa lịch sử chat
    """
    try:
        success = await chat_service.delete_chat_history(session_id, current_user.user_id)
        return {"success": success, "message": "Chat history deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")

@router.get("/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_user)
):
    """
    Lấy danh sách sessions của user
    """
    try:
        sessions = await chat_service.get_user_sessions(current_user.user_id)
        return {"user_id": current_user.user_id, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

#==========================================================================================================================================
# HEALTH CHECK
#==========================================================================================================================================

@router.get("/health")
async def chat_health():
    """
    Health check cho chat system
    """
    try:
        # Test orchestrator connection
        orchestrator = agents_manager.get_conversation_orchestrator()

        return {
            "status": "healthy",
            "conversation_orchestrator": "ready" if orchestrator else "unavailable",
            "endpoints": [
                "/chat/message",
                "/chat/quick",
                "/chat/ws/{session_id}",
                "/chat/history/{session_id}"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }