# api/services/chat_service.py
"""
Chat Service - Business logic cho chat system
Dev2: API Integration & Services - kết nối API với Dev1's agents và Dev3's database
Current: 2025-07-03 13:07:21 UTC, User: xthanh1910
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid

# Import agents từ Dev1 để xử lý conversation logic
from ..agents_manager import agents_manager
from agents.conversational.context_manager_agent import SessionContext
from agents.conversational.query_understanding_agent import QueryUnderstanding

# Import database models/connections từ Dev3 (placeholder imports)
# Dev3 sẽ implement database thực tế
try:
    from database.models.chat_models import ChatMessage, ChatSession
    from database.models.user_models import UserActivity
    from database.connections.chat_db import ChatDatabase
    from database.connections.user_db import UserDatabase
except ImportError:
    # Fallback nếu Dev3 chưa implement
    print("Warning: Database models not found, using mock implementations")
    ChatMessage = dict
    ChatSession = dict
    UserActivity = dict
    ChatDatabase = None
    UserDatabase = None

# Import cache service để lưu session context
from .cache_service import CacheService

class ChatService:
    """
    Service xử lý logic chat
    Dev2 chỉ làm integration - không viết AI logic
    """

    def __init__(self):
        """
        Khởi tạo ChatService
        """
        self.cache_service = CacheService()

        # Database connections (Dev3's responsibility)
        self.chat_db = ChatDatabase() if ChatDatabase else None
        self.user_db = UserDatabase() if UserDatabase else None

        # Mock storage nếu database chưa có
        self._mock_sessions = {}  # session_id -> session_data
        self._mock_messages = {}  # session_id -> [messages]

        # Service configuration
        self.max_session_duration_hours = 24
        self.max_context_turns = 20
        self.default_timeout_seconds = 30

        print(f"[{datetime.utcnow()}] ChatService initialized by user: xthanh1910")

    #======================================================================================================================================
    # MAIN CHAT PROCESSING METHODS
    #======================================================================================================================================

    async def process_user_message(
        self,
        user_message: str,
        session_id: str,
        user_id: str,
        message_metadata: Optional[Dict] = None,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Xử lý message của user
        Dev2: Kết nối user input → Dev1's agents → Dev3's database
        """
        start_time = time.time()

        try:
            # Bước 1: Validate input
            if not user_message or not user_message.strip():
                raise ValueError("Message không được để trống")

            if not session_id:
                session_id = f"session_{user_id}_{int(time.time())}"

            # Bước 2: Lấy hoặc tạo session context
            session_context = await self._get_or_create_session_context(
                session_id=session_id,
                user_id=user_id
            )

            # Bước 3: Cập nhật session với message mới
            await self._add_user_message_to_context(
                session_context=session_context,
                user_message=user_message,
                metadata=message_metadata or {}
            )

            # Bước 4: Gọi conversation orchestrator từ Dev1
            orchestrator = agents_manager.get_conversation_orchestrator()

            if streaming:
                # Streaming response
                result = orchestrator.process_conversation_stream(
                    user_query=user_message,
                    session_context=session_context,
                    user_id=user_id
                )
            else:
                # Regular response
                result = orchestrator.process_conversation(
                    user_query=user_message,
                    session_context=session_context,
                    user_id=user_id
                )

            # Bước 5: Kiểm tra kết quả từ agents Dev1
            if result.status == "error":
                error_response = {
                    "success": False,
                    "error": result.error_message,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time
                }

                # Log error vào database (Dev3)
                await self._log_chat_error(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    error=result.error_message
                )

                return error_response

            # Bước 6: Extract response từ agents result
            flow_data = result.result
            final_response = flow_data.get("final_response", {})
            ai_response = final_response.get("main_answer", "Xin lỗi, tôi gặp vấn đề khi xử lý câu hỏi.")

            # Bước 7: Cập nhật session context với AI response
            await self._add_ai_response_to_context(
                session_context=session_context,
                ai_response=ai_response,
                agent_metadata=flow_data
            )

            # Bước 8: Lưu conversation vào database (Dev3)
            await self._save_conversation_turn(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                ai_response=ai_response,
                agent_metadata=flow_data,
                processing_time=result.execution_time
            )

            # Bước 9: Cập nhật user activity (Dev3)
            await self._track_user_chat_activity(
                user_id=user_id,
                session_id=session_id,
                message_length=len(user_message),
                response_length=len(ai_response),
                processing_time=result.execution_time
            )

            # Bước 10: Lưu session context vào cache
            await self._save_session_context(session_context)

            processing_time = time.time() - start_time

            # Bước 11: Tạo response cho frontend Dev4
            return {
                "success": True,
                "response": ai_response,
                "session_id": session_id,
                "processing_time": processing_time,
                "agent_info": {
                    "agents_used": len(flow_data.get("agent_executions", [])),
                    "success_rate": flow_data.get("success_rate", 0),
                    "flow_status": flow_data.get("status", "unknown"),
                    "query_understanding": flow_data.get("query_understanding_result", {}),
                    "video_results_count": len(flow_data.get("retrieval_result", {}).get("retrieved_videos", []))
                },
                "media_references": final_response.get("media_references", []),
                "follow_up_suggestions": final_response.get("follow_up_suggestions", []),
                "conversation_context": {
                    "turn_count": len(session_context.conversation_turns),
                    "mentioned_videos": len(session_context.mentioned_videos),
                    "active_entities": len(session_context.active_entities)
                }
            }

        except Exception as e:
            # Log exception và return error response
            await self._log_chat_error(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                error=str(e)
            )

            return {
                "success": False,
                "error": f"Lỗi xử lý chat: {str(e)}",
                "session_id": session_id,
                "processing_time": time.time() - start_time
            }

    async def process_quick_chat(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xử lý quick chat - response nhanh không cần context phức tạp
        """
        start_time = time.time()

        try:
            if not session_id:
                session_id = f"quick_{user_id}_{int(time.time())}"

            # Tạo context đơn giản cho quick chat
            simple_context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                start_time=str(time.time()),
                conversation_turns=[],
                active_entities=[],
                mentioned_videos=[],
                search_history=[]
            )

            # Gọi quick_chat method từ Dev1's orchestrator
            orchestrator = agents_manager.get_conversation_orchestrator()
            result = orchestrator.quick_chat(
                user_query=user_message,
                session_id=session_id,
                user_id=user_id
            )

            if result.status == "error":
                return {
                    "success": False,
                    "error": result.error_message,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time
                }

            flow_data = result.result
            final_response = flow_data.get("final_response", {})
            ai_response = final_response.get("main_answer", "")

            # Lưu quick chat vào database (đơn giản hơn regular chat)
            await self._save_quick_chat(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                ai_response=ai_response,
                processing_time=result.execution_time
            )

            return {
                "success": True,
                "response": ai_response,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "mode": "quick",
                "media_references": final_response.get("media_references", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Quick chat failed: {str(e)}",
                "session_id": session_id or "unknown",
                "processing_time": time.time() - start_time
            }

    #======================================================================================================================================
    # SESSION MANAGEMENT METHODS
    #======================================================================================================================================

    async def _get_or_create_session_context(
        self,
        session_id: str,
        user_id: str
    ) -> SessionContext:
        """
        Lấy session context từ cache hoặc tạo mới
        """
        try:
            # Bước 1: Thử lấy từ cache trước
            cached_context = await self.cache_service.get_session_context(session_id)

            if cached_context:
                # Kiểm tra session có hết hạn không
                start_time = float(cached_context.start_time)
                session_age_hours = (time.time() - start_time) / 3600

                if session_age_hours < self.max_session_duration_hours:
                    return cached_context

            # Bước 2: Thử lấy từ database (Dev3)
            if self.chat_db:
                db_session = await self.chat_db.get_session_context(session_id)
                if db_session:
                    return self._convert_db_session_to_context(db_session)

            # Bước 3: Tạo session context mới
            new_context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                start_time=str(time.time()),
                conversation_turns=[],
                active_entities=[],
                mentioned_videos=[],
                search_history=[]
            )

            # Lưu session mới vào database (Dev3)
            await self._create_new_session_in_db(new_context)

            return new_context

        except Exception as e:
            print(f"Error getting session context: {str(e)}")
            # Fallback: tạo context tạm thời
            return SessionContext(
                session_id=session_id,
                user_id=user_id,
                start_time=str(time.time()),
                conversation_turns=[],
                active_entities=[],
                mentioned_videos=[],
                search_history=[]
            )

    async def _save_session_context(self, context: SessionContext) -> bool:
        """
        Lưu session context vào cache và database
        """
        try:
            # Lưu vào cache để access nhanh
            await self.cache_service.save_session_context(
                session_id=context.session_id,
                context=context,
                ttl=self.max_session_duration_hours * 3600
            )

            # Lưu vào database (Dev3) để persistent storage
            if self.chat_db:
                await self.chat_db.update_session_context(
                    session_id=context.session_id,
                    context_data=context.dict()
                )
            else:
                # Mock storage nếu chưa có database
                self._mock_sessions[context.session_id] = context.dict()

            return True

        except Exception as e:
            print(f"Error saving session context: {str(e)}")
            return False

    async def _add_user_message_to_context(
        self,
        session_context: SessionContext,
        user_message: str,
        metadata: Dict[str, Any]
    ):
        """
        Thêm user message vào session context
        """
        # Tạo conversation turn mới
        turn = {
            "turn_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "user_message_metadata": metadata,
            "ai_response": None,  # Sẽ được fill sau
            "agent_metadata": None
        }

        # Thêm vào context
        session_context.conversation_turns.append(turn)

        # Giới hạn số turns để tránh context quá dài
        if len(session_context.conversation_turns) > self.max_context_turns:
            session_context.conversation_turns = session_context.conversation_turns[-self.max_context_turns:]

    async def _add_ai_response_to_context(
        self,
        session_context: SessionContext,
        ai_response: str,
        agent_metadata: Dict[str, Any]
    ):
        """
        Thêm AI response vào turn cuối của session context
        """
        if session_context.conversation_turns:
            last_turn = session_context.conversation_turns[-1]
            last_turn["ai_response"] = ai_response
            last_turn["agent_metadata"] = agent_metadata

            # Cập nhật entities và videos nếu có
            query_result = agent_metadata.get("query_understanding_result", {})
            if query_result.get("entities"):
                for entity in query_result["entities"]:
                    if entity not in session_context.active_entities:
                        session_context.active_entities.append(entity)

            retrieval_result = agent_metadata.get("retrieval_result", {})
            if retrieval_result.get("retrieved_videos"):
                for video in retrieval_result["retrieved_videos"]:
                    video_id = video.get("video_id")
                    if video_id and video_id not in session_context.mentioned_videos:
                        session_context.mentioned_videos.append(video_id)

    #======================================================================================================================================
    # DATABASE INTERACTION METHODS
    #======================================================================================================================================

    async def _save_conversation_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        agent_metadata: Dict[str, Any],
        processing_time: float
    ):
        """
        Lưu conversation turn vào database (Dev3)
        """
        try:
            # Tạo message data để lưu vào database
            message_data = {
                "session_id": session_id,
                "user_id": user_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "agent_metadata": agent_metadata,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow(),
                "message_type": "regular"
            }

            if self.chat_db:
                # Lưu vào database thật (Dev3)
                await self.chat_db.save_chat_message(message_data)
            else:
                # Mock storage
                if session_id not in self._mock_messages:
                    self._mock_messages[session_id] = []
                self._mock_messages[session_id].append(message_data)

        except Exception as e:
            print(f"Error saving conversation turn: {str(e)}")

    async def _save_quick_chat(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        processing_time: float
    ):
        """
        Lưu quick chat vào database (đơn giản hơn regular chat)
        """
        try:
            message_data = {
                "session_id": session_id,
                "user_id": user_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow(),
                "message_type": "quick"
            }

            if self.chat_db:
                await self.chat_db.save_chat_message(message_data)
            else:
                # Mock storage
                if session_id not in self._mock_messages:
                    self._mock_messages[session_id] = []
                self._mock_messages[session_id].append(message_data)

        except Exception as e:
            print(f"Error saving quick chat: {str(e)}")

    async def _log_chat_error(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        error: str
    ):
        """
        Log lỗi chat vào database để debug
        """
        try:
            error_data = {
                "session_id": session_id,
                "user_id": user_id,
                "user_message": user_message,
                "error": error,
                "timestamp": datetime.utcnow(),
                "logged_by": "chat_service"
            }

            if self.chat_db:
                await self.chat_db.log_chat_error(error_data)
            else:
                print(f"Chat Error: {error_data}")

        except Exception as e:
            print(f"Error logging chat error: {str(e)}")

    async def _track_user_chat_activity(
        self,
        user_id: str,
        session_id: str,
        message_length: int,
        response_length: int,
        processing_time: float
    ):
        """
        Track user activity cho analytics (Dev3)
        """
        try:
            activity_data = {
                "user_id": user_id,
                "activity_type": "chat",
                "session_id": session_id,
                "activity_data": {
                    "message_length": message_length,
                    "response_length": response_length,
                    "processing_time": processing_time
                },
                "timestamp": datetime.utcnow()
            }

            if self.user_db:
                await self.user_db.track_user_activity(activity_data)

        except Exception as e:
            print(f"Error tracking user activity: {str(e)}")

    #======================================================================================================================================
    # PUBLIC API METHODS
    #======================================================================================================================================

    async def get_chat_history(
        self,
        session_id: str,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử chat của session (cho frontend Dev4)
        """
        try:
            if self.chat_db:
                # Lấy từ database (Dev3)
                messages = await self.chat_db.get_chat_history(
                    session_id=session_id,
                    user_id=user_id,
                    limit=limit
                )
            else:
                # Mock storage
                messages = self._mock_messages.get(session_id, [])
                messages = messages[-limit:] if len(messages) > limit else messages

            # Format messages cho frontend
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "timestamp": msg.get("timestamp"),
                    "user_message": msg.get("user_message"),
                    "ai_response": msg.get("ai_response"),
                    "processing_time": msg.get("processing_time", 0),
                    "message_type": msg.get("message_type", "regular")
                })

            return formatted_messages

        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []

    async def delete_chat_history(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        Xóa lịch sử chat của session
        """
        try:
            if self.chat_db:
                # Xóa từ database (Dev3)
                success = await self.chat_db.delete_chat_history(session_id, user_id)
            else:
                # Mock storage
                if session_id in self._mock_messages:
                    del self._mock_messages[session_id]
                if session_id in self._mock_sessions:
                    del self._mock_sessions[session_id]
                success = True

            # Xóa từ cache
            await self.cache_service.delete_session_context(session_id)

            return success

        except Exception as e:
            print(f"Error deleting chat history: {str(e)}")
            return False

    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách sessions của user
        """
        try:
            if self.chat_db:
                sessions = await self.chat_db.get_user_sessions(user_id)
            else:
                # Mock storage
                sessions = []
                for session_id, session_data in self._mock_sessions.items():
                    if session_data.get("user_id") == user_id:
                        message_count = len(self._mock_messages.get(session_id, []))
                        sessions.append({
                            "session_id": session_id,
                            "start_time": session_data.get("start_time"),
                            "message_count": message_count,
                            "last_activity": session_data.get("last_activity", session_data.get("start_time"))
                        })

            return sessions

        except Exception as e:
            print(f"Error getting user sessions: {str(e)}")
            return []

    #======================================================================================================================================
    # HELPER METHODS
    #======================================================================================================================================

    def _convert_db_session_to_context(self, db_session) -> SessionContext:
        """
        Convert database session data thành SessionContext object
        """
        return SessionContext(
            session_id=db_session.get("session_id"),
            user_id=db_session.get("user_id"),
            start_time=db_session.get("start_time"),
            conversation_turns=db_session.get("conversation_turns", []),
            active_entities=db_session.get("active_entities", []),
            mentioned_videos=db_session.get("mentioned_videos", []),
            search_history=db_session.get("search_history", [])
        )

    async def _create_new_session_in_db(self, context: SessionContext):
        """
        Tạo session mới trong database (Dev3)
        """
        try:
            if self.chat_db:
                await self.chat_db.create_chat_session({
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "start_time": context.start_time,
                    "status": "active"
                })
            else:
                # Mock storage
                self._mock_sessions[context.session_id] = context.dict()

        except Exception as e:
            print(f"Error creating session in DB: {str(e)}")

    #======================================================================================================================================
    # HEALTH CHECK & SERVICE MANAGEMENT
    #======================================================================================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check cho ChatService
        """
        try:
            start_time = time.time()

            # Test agents connection (Dev1)
            orchestrator = agents_manager.get_conversation_orchestrator()
            agents_ok = orchestrator is not None

            # Test database connection (Dev3)
            db_ok = True
            if self.chat_db:
                try:
                    db_ok = await self.chat_db.health_check()
                except Exception:
                    db_ok = False

            # Test cache connection
            cache_ok = await self.cache_service.health_check()
            cache_status = cache_ok.get("status") == "healthy"

            response_time = (time.time() - start_time) * 1000
            overall_status = "healthy" if (agents_ok and db_ok and cache_status) else "degraded"

            return {
                "status": overall_status,
                "service": "chat_service",
                "response_time_ms": round(response_time, 2),
                "components": {
                    "conversation_orchestrator": "healthy" if agents_ok else "error",
                    "database": "healthy" if db_ok else "error",
                    "cache": "healthy" if cache_status else "error"
                },
                "metrics": {
                    "active_sessions": len(self._mock_sessions),
                    "total_messages": sum(len(msgs) for msgs in self._mock_messages.values()),
                    "max_session_duration_hours": self.max_session_duration_hours,
                    "max_context_turns": self.max_context_turns
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def initialize(self):
        """
        Khởi tạo ChatService khi app startup
        """
        try:
            # Initialize database connections (Dev3)
            if self.chat_db:
                await self.chat_db.connect()
            if self.user_db:
                await self.user_db.connect()

            # Initialize cache service
            await self.cache_service.initialize()

            print(f"[{datetime.utcnow()}] ChatService initialized successfully")

        except Exception as e:
            print(f"[{datetime.utcnow()}] ChatService initialization failed: {str(e)}")
            raise

    async def shutdown(self):
        """
        Graceful shutdown ChatService
        """
        try:
            # Close database connections (Dev3)
            if self.chat_db:
                await self.chat_db.disconnect()
            if self.user_db:
                await self.user_db.disconnect()

            # Close cache service
            await self.cache_service.close()

            print(f"[{datetime.utcnow()}] ChatService shutdown completed")

        except Exception as e:
            print(f"[{datetime.utcnow()}] ChatService shutdown error: {str(e)}")

    #======================================================================================================================================
    # ADMIN/ANALYTICS METHODS
    #======================================================================================================================================

    async def get_chat_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê chat cho admin dashboard
        """
        try:
            if self.chat_db:
                stats = await self.chat_db.get_chat_statistics()
            else:
                # Mock statistics
                total_sessions = len(self._mock_sessions)
                total_messages = sum(len(msgs) for msgs in self._mock_messages.values())

                stats = {
                    "total_sessions": total_sessions,
                    "total_messages": total_messages,
                    "active_sessions": total_sessions,  # Mock: assume all active
                    "average_messages_per_session": total_messages / max(total_sessions, 1),
                    "total_users": len(set(session.get("user_id") for session in self._mock_sessions.values()))
                }

            return stats

        except Exception as e:
            return {"error": str(e)}

    async def cleanup_old_sessions(self, hours_old: int = 48):
        """
        Dọn dẹp sessions cũ (background task)
        """
        try:
            cutoff_time = time.time() - (hours_old * 3600)
            cleaned_count = 0

            if self.chat_db:
                cleaned_count = await self.chat_db.cleanup_old_sessions(cutoff_time)
            else:
                # Mock cleanup
                sessions_to_remove = []
                for session_id, session_data in self._mock_sessions.items():
                    if float(session_data.get("start_time", 0)) < cutoff_time:
                        sessions_to_remove.append(session_id)

                for session_id in sessions_to_remove:
                    del self._mock_sessions[session_id]
                    if session_id in self._mock_messages:
                        del self._mock_messages[session_id]

                cleaned_count = len(sessions_to_remove)

            print(f"[{datetime.utcnow()}] Cleaned up {cleaned_count} old chat sessions")
            return cleaned_count

        except Exception as e:
            print(f"Error cleaning up old sessions: {str(e)}")
            return 0

# Export service instance
chat_service = ChatService()