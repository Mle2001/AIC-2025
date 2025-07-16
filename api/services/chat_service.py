from api.models.chat_models import ChatRequest
from api.services.video_service import VideoService
from api.config import get_openai_api_key, settings
from typing import AsyncGenerator, List, Dict, Any
import asyncio
import uuid
import os
import requests
import json
import logging

# Placeholder for chat service
class ChatService:
    def __init__(self):
        # Loại bỏ orchestrator vì không còn sử dụng agents
        self.session_data = {}
        self.video_service = VideoService()

    async def process_message(self, request: ChatRequest, user_id: str) -> Dict[str, Any]:
        # Sử dụng API key từ config nếu không có trong request
        api_key = request.openai_api_key or get_openai_api_key()
        
        # Kiểm tra nếu có video_id trong request để query video
        if hasattr(request, 'video_id') and request.video_id:
            try:
                # Query video với VideoRAG
                video_result = await self.video_service.query_video(
                    request.video_id, 
                    request.message, 
                    api_key
                )
                
                # Lấy thông tin video để trả về
                video_info = await self.video_service.get_video_info(request.video_id)
                
                return {
                    "response": video_result["response"],
                    "videos": [{
                        "id": request.video_id, 
                        "query": request.message,
                        "name": video_info.get("original_name", "Unknown"),
                        "status": video_info.get("status", "unknown")
                    }],
                    "session_id": request.session_id or str(uuid.uuid4()),
                    "model": request.model or settings.openai_model,
                    "usage": None,
                    "source": "videorag"
                }
            except Exception as e:
                logging.error(f"VideoRAG error: {e}")
                return {
                    "response": f"VideoRAG error: {str(e)}",
                    "videos": [],
                    "session_id": request.session_id or str(uuid.uuid4()),
                    "model": request.model or settings.openai_model,
                    "usage": None,
                    "source": "error"
                }
        
        # Nếu có API key thì gọi OpenAI API
        if api_key:
            # Ví dụ gọi OpenAI Chat Completion API
            openai_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.model or settings.openai_model,
                "messages": [{"role": "user", "content": request.message}],
                "temperature": request.temperature or 0.7
            }
            try:
                resp = requests.post(openai_url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                ai_message = data["choices"][0]["message"]["content"]
                usage = data.get("usage")
                return {
                    "response": ai_message,
                    "videos": [],
                    "session_id": request.session_id or str(uuid.uuid4()),
                    "model": request.model or settings.openai_model,
                    "usage": usage,
                    "source": "openai"
                }
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                return {
                    "response": f"OpenAI API error: {str(e)}",
                    "videos": [],
                    "session_id": request.session_id or str(uuid.uuid4()),
                    "model": request.model or settings.openai_model,
                    "usage": None,
                    "source": "error"
                }
        
        # Fallback demo response
        return {
            "response": f"Demo AI response for user {user_id}. Please configure OpenAI API key in backend.",
            "videos": [],
            "session_id": getattr(request, 'session_id', str(uuid.uuid4())),
            "model": request.model or settings.openai_model,
            "usage": None,
            "source": "demo"
        }

    async def get_video_list(self, user_id: str) -> List[Dict[str, Any]]:
        """Lấy danh sách video của user"""
        try:
            return await self.video_service.list_videos(user_id)
        except Exception as e:
            logging.error(f"Error getting video list: {e}")
            return []

    async def stream_response(self, request, user_id: str) -> AsyncGenerator[str, None]:
        # Dummy streaming generator
        for chunk in ["Đây ", "là ", "phản ", "hồi ", "từng ", "phần"]:
            yield chunk
            await asyncio.sleep(0.2)

    async def manage_session(self, session_id: str, user_id: str):
        # Dummy session management
        return {"session_id": session_id, "user_id": user_id}
