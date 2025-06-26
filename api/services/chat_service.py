from agents.orchestrator.conversation_orchestrator import ConversationOrchestrator
#from database.models.user_session import UserSession
#from database.models.conversation import Conversation
from api.models.chat_models import ChatRequest
from typing import AsyncGenerator
import asyncio
import uuid
import os
import requests

# Placeholder for chat service
class ChatService:
    def __init__(self):
        self.orchestrator = ConversationOrchestrator()

    async def process_message(self, request: ChatRequest, user_id: str):
        # Nếu có openai_api_key thì gọi OpenAI API, nếu không thì trả về demo
        if request.openai_api_key:
            # Ví dụ gọi OpenAI Chat Completion API
            openai_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {request.openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.model or "gpt-3.5-turbo",
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
                    "model": request.model,
                    "usage": usage
                }
            except Exception as e:
                return {
                    "response": f"OpenAI API error: {str(e)}",
                    "videos": [],
                    "session_id": request.session_id or str(uuid.uuid4()),
                    "model": request.model,
                    "usage": None
                }
        return {
            "response": f"Demo AI response for user {user_id}",
            "videos": [],
            "session_id": getattr(request, 'session_id', str(uuid.uuid4())),
            "model": request.model,
            "usage": None
        }

    async def stream_response(self, request, user_id: str) -> AsyncGenerator[str, None]:
        # Dummy streaming generator
        for chunk in ["Đây ", "là ", "phản ", "hồi ", "từng ", "phần"]:
            yield chunk
            await asyncio.sleep(0.2)

    async def manage_session(self, session_id: str, user_id: str):
        # Dummy session management
        return {"session_id": session_id, "user_id": user_id}
