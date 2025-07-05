# api/agents_manager.py
"""
Agents Manager - Quản lý và kết nối với Dev1's AI Agents
Dev2: API Integration Layer - bridge giữa API và AI Agents từ Dev1
Current: 2025-07-03 14:30:41 UTC, User: xthanh1910
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Import cache service để cache agent results
from .services.cache_service import CacheService

#======================================================================================================================================
# IMPORT DEV1'S AGENTS (placeholder imports)
# Dev1 sẽ implement các agents thực tế
#======================================================================================================================================

try:
    # Conversation Agents
    from agents.conversational.conversation_orchestrator import ConversationOrchestrator
    from agents.conversational.context_manager_agent import ContextManagerAgent, SessionContext
    from agents.conversational.query_understanding_agent import QueryUnderstandingAgent
    from agents.conversational.response_synthesis_agent import ResponseSynthesisAgent

    # Video Processing Agents
    from agents.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator, PreprocessingConfig
    from agents.extraction.video_frame_extraction_agent import VideoFrameExtractionAgent
    from agents.extraction.audio_extraction_agent import AudioExtractionAgent
    from agents.extraction.speech_to_text_agent import SpeechToTextAgent

    # Content Analysis Agents
    from agents.analysis.content_analysis_agent import ContentAnalysisAgent
    from agents.analysis.video_understanding_agent import VideoUnderstandingAgent
    from agents.analysis.multimodal_analysis_agent import MultimodalAnalysisAgent

    # Search and Retrieval Agents
    from agents.retrieval.video_retrieval_agent import VideoRetrievalAgent
    from agents.retrieval.semantic_search_agent import SemanticSearchAgent
    from agents.retrieval.content_matching_agent import ContentMatchingAgent

    # Knowledge Graph Agents
    from agents.knowledge.knowledge_graph_agent import KnowledgeGraphAgent
    from agents.knowledge.entity_extraction_agent import EntityExtractionAgent
    from agents.knowledge.relationship_mapping_agent import RelationshipMappingAgent

    AGENTS_AVAILABLE = True
    print(f"[{datetime.utcnow()}] All Dev1 agents imported successfully")

except ImportError as e:
    # Fallback khi Dev1 chưa implement agents
    print(f"[{datetime.utcnow()}] Warning: Dev1 agents not found, using mock implementations")
    print(f"Import error: {str(e)}")
    AGENTS_AVAILABLE = False

    # Mock classes để API không crash
    class MockAgent:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.status = "mock"

        def process(self, *args, **kwargs):
            return {
                "status": "success",
                "result": f"Mock result from {self.agent_name}",
                "execution_time": 1.0,
                "mock": True
            }

    # Mock imports
    ConversationOrchestrator = MockAgent
    PreprocessingOrchestrator = MockAgent
    SessionContext = dict
    PreprocessingConfig = dict

#======================================================================================================================================
# AGENT STATUS ENUMS
#======================================================================================================================================

class AgentStatus(str, Enum):
    """
    Trạng thái của agents
    """
    AVAILABLE = "available"       # Sẵn sàng xử lý
    BUSY = "busy"                # Đang xử lý task
    ERROR = "error"              # Gặp lỗi
    MAINTENANCE = "maintenance"   # Đang bảo trì
    DISABLED = "disabled"        # Bị vô hiệu hóa

class TaskPriority(str, Enum):
    """
    Độ ưu tiên của task
    """
    LOW = "low"                  # Ưu tiên thấp
    NORMAL = "normal"            # Ưu tiên bình thường
    HIGH = "high"                # Ưu tiên cao
    URGENT = "urgent"            # Khẩn cấp
    CRITICAL = "critical"        # Cực kỳ quan trọng

#======================================================================================================================================
# AGENT RESULT CLASSES
#======================================================================================================================================

@dataclass
class AgentResult:
    """
    Kết quả từ agent execution
    """
    status: str                          # "success", "error", "timeout"
    result: Optional[Dict[str, Any]]     # Kết quả thực tế
    error_message: Optional[str]         # Thông báo lỗi
    execution_time: float                # Thời gian thực thi (seconds)
    agent_name: str                      # Tên agent đã thực thi
    task_id: Optional[str] = None        # ID của task
    metadata: Optional[Dict[str, Any]] = None  # Metadata bổ sung

@dataclass
class AgentPoolStats:
    """
    Thống kê agent pool
    """
    total_agents: int
    available_agents: int
    busy_agents: int
    error_agents: int
    total_tasks_processed: int
    avg_execution_time: float
    success_rate: float

#======================================================================================================================================
# AGENT POOL MANAGER
#======================================================================================================================================

class AgentPoolManager:
    """
    Quản lý pool của agents để xử lý concurrent tasks
    """

    def __init__(self):
        self.cache = CacheService()

        # Agent instances
        self.agents = {}
        self.agent_status = {}
        self.agent_stats = {}

        # Task management
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.active_tasks = {}
        self.task_results = {}

        # Thread pool cho blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

        # Configuration
        self.max_concurrent_tasks = 20
        self.task_timeout_seconds = 300  # 5 minutes
        self.agent_health_check_interval = 60  # 1 minute

        # Background tasks
        self.background_tasks = []

        print(f"[{datetime.utcnow()}] AgentPoolManager initialized by user: xthanh1910")

    async def initialize(self):
        """
        Khởi tạo agent pool
        """
        try:
            # Bước 1: Initialize cache service
            await self.cache.initialize()

            # Bước 2: Initialize agents
            await self._initialize_agents()

            # Bước 3: Start background tasks
            await self._start_background_tasks()

            print(f"[{datetime.utcnow()}] Agent pool initialized successfully")

        except Exception as e:
            print(f"[{datetime.utcnow()}] Agent pool initialization failed: {str(e)}")
            raise

    async def _initialize_agents(self):
        """
        Khởi tạo tất cả agents
        """
        try:
            if AGENTS_AVAILABLE:
                # Initialize conversation agents
                self.agents['conversation_orchestrator'] = ConversationOrchestrator()
                self.agents['context_manager'] = ContextManagerAgent()
                self.agents['query_understanding'] = QueryUnderstandingAgent()
                self.agents['response_synthesis'] = ResponseSynthesisAgent()

                # Initialize video processing agents
                self.agents['preprocessing_orchestrator'] = PreprocessingOrchestrator()
                self.agents['video_frame_extraction'] = VideoFrameExtractionAgent()
                self.agents['audio_extraction'] = AudioExtractionAgent()
                self.agents['speech_to_text'] = SpeechToTextAgent()

                # Initialize content analysis agents
                self.agents['content_analysis'] = ContentAnalysisAgent()
                self.agents['video_understanding'] = VideoUnderstandingAgent()
                self.agents['multimodal_analysis'] = MultimodalAnalysisAgent()

                # Initialize retrieval agents
                self.agents['video_retrieval'] = VideoRetrievalAgent()
                self.agents['semantic_search'] = SemanticSearchAgent()
                self.agents['content_matching'] = ContentMatchingAgent()

                # Initialize knowledge graph agents
                self.agents['knowledge_graph'] = KnowledgeGraphAgent()
                self.agents['entity_extraction'] = EntityExtractionAgent()
                self.agents['relationship_mapping'] = RelationshipMappingAgent()

            else:
                # Mock agents nếu Dev1 chưa implement
                agent_names = [
                    'conversation_orchestrator', 'context_manager', 'query_understanding',
                    'response_synthesis', 'preprocessing_orchestrator', 'video_frame_extraction',
                    'audio_extraction', 'speech_to_text', 'content_analysis',
                    'video_understanding', 'multimodal_analysis', 'video_retrieval',
                    'semantic_search', 'content_matching', 'knowledge_graph',
                    'entity_extraction', 'relationship_mapping'
                ]

                for agent_name in agent_names:
                    self.agents[agent_name] = MockAgent(agent_name)

            # Initialize agent status
            for agent_name in self.agents:
                self.agent_status[agent_name] = AgentStatus.AVAILABLE
                self.agent_stats[agent_name] = {
                    'total_tasks': 0,
                    'successful_tasks': 0,
                    'failed_tasks': 0,
                    'total_execution_time': 0.0,
                    'last_used': None,
                    'error_count': 0
                }

            print(f"[{datetime.utcnow()}] Initialized {len(self.agents)} agents")

        except Exception as e:
            print(f"Error initializing agents: {str(e)}")
            raise

    async def _start_background_tasks(self):
        """
        Start background monitoring tasks
        """
        try:
            # Task processor
            task_processor = asyncio.create_task(self._process_task_queue())
            self.background_tasks.append(task_processor)

            # Health checker
            health_checker = asyncio.create_task(self._periodic_health_check())
            self.background_tasks.append(health_checker)

            # Stats updater
            stats_updater = asyncio.create_task(self._update_stats_periodically())
            self.background_tasks.append(stats_updater)

            print(f"[{datetime.utcnow()}] Started {len(self.background_tasks)} background tasks")

        except Exception as e:
            print(f"Error starting background tasks: {str(e)}")

    async def _process_task_queue(self):
        """
        Background task để process task queue
        """
        while True:
            try:
                # Get task from queue với timeout
                task_data = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # Process task
                await self._execute_task(task_data)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                print(f"Error processing task queue: {str(e)}")
                await asyncio.sleep(1)

    async def _execute_task(self, task_data: Dict[str, Any]):
        """
        Execute một task
        """
        task_id = task_data.get('task_id')
        agent_name = task_data.get('agent_name')
        method_name = task_data.get('method_name', 'process')
        args = task_data.get('args', ())
        kwargs = task_data.get('kwargs', {})

        try:
            # Mark task as active
            self.active_tasks[task_id] = {
                'agent_name': agent_name,
                'started_at': time.time(),
                'status': 'running'
            }

            # Set agent status to busy
            self.agent_status[agent_name] = AgentStatus.BUSY

            # Execute agent method
            start_time = time.time()
            agent = self.agents[agent_name]

            if hasattr(agent, method_name):
                method = getattr(agent, method_name)

                # Execute method (sync hoặc async)
                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    # Run blocking method trong thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: method(*args, **kwargs)
                    )

                execution_time = time.time() - start_time

                # Create successful result
                agent_result = AgentResult(
                    status="success",
                    result=result,
                    error_message=None,
                    execution_time=execution_time,
                    agent_name=agent_name,
                    task_id=task_id
                )

                # Update stats
                self._update_agent_stats(agent_name, True, execution_time)

            else:
                raise AttributeError(f"Agent {agent_name} doesn't have method {method_name}")

        except Exception as e:
            execution_time = time.time() - start_time

            # Create error result
            agent_result = AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=execution_time,
                agent_name=agent_name,
                task_id=task_id
            )

            # Update stats
            self._update_agent_stats(agent_name, False, execution_time)

            print(f"Task execution error: {str(e)}")

        finally:
            # Clean up
            self.task_results[task_id] = agent_result

            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            # Set agent back to available
            self.agent_status[agent_name] = AgentStatus.AVAILABLE

    def _update_agent_stats(self, agent_name: str, success: bool, execution_time: float):
        """
        Cập nhật thống kê agent
        """
        try:
            stats = self.agent_stats[agent_name]
            stats['total_tasks'] += 1
            stats['total_execution_time'] += execution_time
            stats['last_used'] = time.time()

            if success:
                stats['successful_tasks'] += 1
            else:
                stats['failed_tasks'] += 1
                stats['error_count'] += 1

        except Exception as e:
            print(f"Error updating agent stats: {str(e)}")

    async def _periodic_health_check(self):
        """
        Định kỳ check health của agents
        """
        while True:
            try:
                await asyncio.sleep(self.agent_health_check_interval)

                for agent_name, agent in self.agents.items():
                    try:
                        # Basic health check
                        if hasattr(agent, 'health_check'):
                            health_result = agent.health_check()
                            if not health_result:
                                self.agent_status[agent_name] = AgentStatus.ERROR

                        # Check if agent has too many errors
                        stats = self.agent_stats[agent_name]
                        if stats['error_count'] > 10:  # More than 10 errors
                            print(f"[{datetime.utcnow()}] Agent {agent_name} has high error count: {stats['error_count']}")

                    except Exception as e:
                        print(f"Health check error for agent {agent_name}: {str(e)}")
                        self.agent_status[agent_name] = AgentStatus.ERROR

            except Exception as e:
                print(f"Periodic health check error: {str(e)}")

    async def _update_stats_periodically(self):
        """
        Định kỳ update và cache stats
        """
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Calculate pool stats
                pool_stats = self.get_pool_stats()

                # Cache stats
                await self.cache.set(
                    "agent_pool_stats",
                    pool_stats.__dict__,
                    ttl=600,  # 10 minutes
                    namespace='system'
                )

            except Exception as e:
                print(f"Stats update error: {str(e)}")

    #======================================================================================================================================
    # PUBLIC METHODS
    #======================================================================================================================================

    async def execute_agent_task(
        self,
        agent_name: str,
        method_name: str = 'process',
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: int = None
    ) -> AgentResult:
        """
        Execute agent task với queue management
        """
        try:
            kwargs = kwargs or {}
            timeout = timeout or self.task_timeout_seconds

            # Kiểm tra agent tồn tại
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")

            # Kiểm tra agent status
            if self.agent_status[agent_name] in [AgentStatus.ERROR, AgentStatus.DISABLED]:
                raise RuntimeError(f"Agent {agent_name} is not available: {self.agent_status[agent_name]}")

            # Tạo task
            task_id = f"task_{agent_name}_{int(time.time())}_{id(args)}"

            task_data = {
                'task_id': task_id,
                'agent_name': agent_name,
                'method_name': method_name,
                'args': args,
                'kwargs': kwargs,
                'priority': priority,
                'created_at': time.time()
            }

            # Add to queue
            await self.task_queue.put(task_data)

            # Wait for result với timeout
            start_wait = time.time()
            while task_id not in self.task_results:
                if time.time() - start_wait > timeout:
                    # Cleanup task nếu timeout
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]

                    return AgentResult(
                        status="timeout",
                        result=None,
                        error_message=f"Task timeout after {timeout} seconds",
                        execution_time=timeout,
                        agent_name=agent_name,
                        task_id=task_id
                    )

                await asyncio.sleep(0.1)  # Check every 100ms

            # Get result
            result = self.task_results[task_id]

            # Cleanup
            del self.task_results[task_id]

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name=agent_name
            )

    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """
        Lấy status của agent
        """
        return self.agent_status.get(agent_name)

    def get_agent_stats(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Lấy stats của agent
        """
        if agent_name not in self.agent_stats:
            return None

        stats = self.agent_stats[agent_name].copy()

        # Calculate derived stats
        if stats['total_tasks'] > 0:
            stats['success_rate'] = stats['successful_tasks'] / stats['total_tasks']
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_tasks']
        else:
            stats['success_rate'] = 0.0
            stats['avg_execution_time'] = 0.0

        return stats

    def get_pool_stats(self) -> AgentPoolStats:
        """
        Lấy thống kê toàn bộ agent pool
        """
        try:
            total_agents = len(self.agents)
            available_agents = sum(1 for status in self.agent_status.values() if status == AgentStatus.AVAILABLE)
            busy_agents = sum(1 for status in self.agent_status.values() if status == AgentStatus.BUSY)
            error_agents = sum(1 for status in self.agent_status.values() if status == AgentStatus.ERROR)

            # Calculate aggregate stats
            total_tasks = sum(stats['total_tasks'] for stats in self.agent_stats.values())
            successful_tasks = sum(stats['successful_tasks'] for stats in self.agent_stats.values())
            total_execution_time = sum(stats['total_execution_time'] for stats in self.agent_stats.values())

            avg_execution_time = total_execution_time / max(total_tasks, 1)
            success_rate = successful_tasks / max(total_tasks, 1)

            return AgentPoolStats(
                total_agents=total_agents,
                available_agents=available_agents,
                busy_agents=busy_agents,
                error_agents=error_agents,
                total_tasks_processed=total_tasks,
                avg_execution_time=avg_execution_time,
                success_rate=success_rate
            )

        except Exception as e:
            print(f"Error calculating pool stats: {str(e)}")
            return AgentPoolStats(0, 0, 0, 0, 0, 0.0, 0.0)

    async def reset_agent(self, agent_name: str) -> bool:
        """
        Reset agent (clear stats, reset status)
        """
        try:
            if agent_name not in self.agents:
                return False

            # Reset stats
            self.agent_stats[agent_name] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'total_execution_time': 0.0,
                'last_used': None,
                'error_count': 0
            }

            # Reset status
            self.agent_status[agent_name] = AgentStatus.AVAILABLE

            print(f"[{datetime.utcnow()}] Agent {agent_name} reset successfully")
            return True

        except Exception as e:
            print(f"Error resetting agent {agent_name}: {str(e)}")
            return False

    async def shutdown(self):
        """
        Graceful shutdown của agent pool
        """
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()

            # Wait for active tasks to complete (với timeout)
            if self.active_tasks:
                print(f"[{datetime.utcnow()}] Waiting for {len(self.active_tasks)} active tasks to complete...")

                timeout = 30  # 30 seconds timeout
                start_time = time.time()

                while self.active_tasks and (time.time() - start_time) < timeout:
                    await asyncio.sleep(1)

                if self.active_tasks:
                    print(f"[{datetime.utcnow()}] Force stopping {len(self.active_tasks)} remaining tasks")

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            # Close cache
            await self.cache.close()

            print(f"[{datetime.utcnow()}] Agent pool shutdown completed")

        except Exception as e:
            print(f"Error during agent pool shutdown: {str(e)}")

#======================================================================================================================================
# MAIN AGENTS MANAGER CLASS
#======================================================================================================================================

class AgentsManager:
    """
    Main manager để interact với tất cả agents
    Dev2: Simplified interface cho services sử dụng
    """

    def __init__(self):
        self.pool_manager = AgentPoolManager()
        self.cache = CacheService()
        self.initialized = False

        print(f"[{datetime.utcnow()}] AgentsManager created for user: xthanh1910")

    async def initialize(self):
        """
        Initialize agents manager
        """
        try:
            if not self.initialized:
                await self.pool_manager.initialize()
                await self.cache.initialize()
                self.initialized = True
                print(f"[{datetime.utcnow()}] AgentsManager initialized successfully")
        except Exception as e:
            print(f"AgentsManager initialization failed: {str(e)}")
            raise

    #======================================================================================================================================
    # CONVERSATION AGENTS
    #======================================================================================================================================

    def get_conversation_orchestrator(self):
        """
        Lấy conversation orchestrator cho chat service
        """
        if not AGENTS_AVAILABLE:
            return MockConversationOrchestrator()

        return ConversationOrchestratorWrapper(self.pool_manager)

    async def process_user_query(
        self,
        user_query: str,
        session_context: Dict[str, Any],
        user_id: str
    ) -> AgentResult:
        """
        Process user query through conversation flow
        """
        try:
            await self._ensure_initialized()

            # Cache key cho query result
            cache_key = f"query_result:{hash(user_query)}:{user_id}"
            cached_result = await self.cache.get(cache_key, namespace='api')

            if cached_result:
                print(f"[{datetime.utcnow()}] Using cached query result for user: {user_id}")
                return AgentResult(**cached_result)

            # Execute through conversation orchestrator
            result = await self.pool_manager.execute_agent_task(
                agent_name='conversation_orchestrator',
                method_name='process_conversation',
                kwargs={
                    'user_query': user_query,
                    'session_context': session_context,
                    'user_id': user_id
                },
                timeout=60  # 1 minute timeout cho conversation
            )

            # Cache successful results
            if result.status == "success":
                await self.cache.set(
                    cache_key,
                    result.__dict__,
                    ttl=300,  # 5 minutes cache
                    namespace='api'
                )

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name="conversation_orchestrator"
            )

    #======================================================================================================================================
    # VIDEO PROCESSING AGENTS
    #======================================================================================================================================

    def get_preprocessing_orchestrator(self):
        """
        Lấy preprocessing orchestrator cho video service
        """
        if not AGENTS_AVAILABLE:
            return MockPreprocessingOrchestrator()

        return PreprocessingOrchestratorWrapper(self.pool_manager)

    async def process_video(
        self,
        video_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Process video through preprocessing pipeline
        """
        try:
            await self._ensure_initialized()

            result = await self.pool_manager.execute_agent_task(
                agent_name='preprocessing_orchestrator',
                method_name='process_video',
                kwargs={
                    'video_path': video_path,
                    'config': config
                },
                timeout=600  # 10 minutes timeout cho video processing
            )

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name="preprocessing_orchestrator"
            )

    #======================================================================================================================================
    # SEARCH & RETRIEVAL AGENTS
    #======================================================================================================================================

    async def search_videos(
        self,
        query: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Search videos using semantic search
        """
        try:
            await self._ensure_initialized()

            # Cache key cho search results
            cache_key = f"search_result:{hash(query)}:{hash(str(filters))}"
            cached_result = await self.cache.get(cache_key, namespace='api')

            if cached_result:
                print(f"[{datetime.utcnow()}] Using cached search result for query: {query}")
                return AgentResult(**cached_result)

            result = await self.pool_manager.execute_agent_task(
                agent_name='video_retrieval',
                method_name='search_videos',
                kwargs={
                    'query': query,
                    'user_id': user_id,
                    'filters': filters or {}
                },
                timeout=30  # 30 seconds timeout cho search
            )

            # Cache successful search results
            if result.status == "success":
                await self.cache.set(
                    cache_key,
                    result.__dict__,
                    ttl=600,  # 10 minutes cache
                    namespace='api'
                )

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name="video_retrieval"
            )

    #======================================================================================================================================
    # AGENT MANAGEMENT METHODS
    #======================================================================================================================================

    async def get_all_agents_status(self) -> Dict[str, Any]:
        """
        Lấy status của tất cả agents
        """
        try:
            await self._ensure_initialized()

            agents_status = {}

            for agent_name in self.pool_manager.agents:
                agents_status[agent_name] = {
                    'status': self.pool_manager.get_agent_status(agent_name),
                    'stats': self.pool_manager.get_agent_stats(agent_name)
                }

            return {
                'agents': agents_status,
                'pool_stats': self.pool_manager.get_pool_stats().__dict__,
                'active_tasks': len(self.pool_manager.active_tasks),
                'queue_size': self.pool_manager.task_queue.qsize(),
                'agents_available': AGENTS_AVAILABLE
            }

        except Exception as e:
            return {'error': str(e)}

    async def reset_agent(self, agent_name: str) -> bool:
        """
        Reset specific agent
        """
        try:
            await self._ensure_initialized()
            return await self.pool_manager.reset_agent(agent_name)
        except Exception as e:
            print(f"Error resetting agent {agent_name}: {str(e)}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check cho toàn bộ agents system
        """
        try:
            start_time = time.time()

            # Check pool manager
            pool_stats = self.pool_manager.get_pool_stats()

            # Check cache connectivity
            cache_result = await self.cache.health_check()
            cache_ok = cache_result.get("status") == "healthy"

            # Test agent execution
            test_result = await self.pool_manager.execute_agent_task(
                agent_name='query_understanding',
                method_name='process' if AGENTS_AVAILABLE else 'process',
                kwargs={'query': 'test health check'},
                timeout=10
            )

            agents_ok = test_result.status in ["success", "error"]  # Either is fine for health check

            response_time = (time.time() - start_time) * 1000

            return {
                'status': 'healthy' if (cache_ok and agents_ok) else 'degraded',
                'service': 'agents_manager',
                'response_time_ms': round(response_time, 2),
                'components': {
                    'agent_pool': 'healthy' if pool_stats.available_agents > 0 else 'error',
                    'cache_service': 'healthy' if cache_ok else 'error',
                    'agents_execution': 'healthy' if agents_ok else 'error'
                },
                'pool_stats': pool_stats.__dict__,
                'agents_available': AGENTS_AVAILABLE,
                'total_agents': len(self.pool_manager.agents),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def _ensure_initialized(self):
        """
        Đảm bảo manager đã được initialized
        """
        if not self.initialized:
            await self.initialize()

    async def shutdown(self):
        """
        Graceful shutdown
        """
        try:
            if self.initialized:
                await self.pool_manager.shutdown()
                await self.cache.close()
                self.initialized = False
                print(f"[{datetime.utcnow()}] AgentsManager shutdown completed")
        except Exception as e:
            print(f"AgentsManager shutdown error: {str(e)}")

#======================================================================================================================================
# WRAPPER CLASSES
#======================================================================================================================================

class ConversationOrchestratorWrapper:
    """
    Wrapper cho ConversationOrchestrator để sử dụng pool manager
    """

    def __init__(self, pool_manager: AgentPoolManager):
        self.pool_manager = pool_manager

    def process_conversation(self, user_query: str, session_context: Dict, user_id: str):
        """
        Sync wrapper cho async agent execution
        """
        try:
            # Tạo event loop nếu cần
            loop = asyncio.get_event_loop()

            # Execute agent task
            result = loop.run_until_complete(
                self.pool_manager.execute_agent_task(
                    agent_name='conversation_orchestrator',
                    method_name='process_conversation',
                    kwargs={
                        'user_query': user_query,
                        'session_context': session_context,
                        'user_id': user_id
                    }
                )
            )

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name="conversation_orchestrator"
            )

    def quick_chat(self, user_query: str, session_id: str, user_id: str):
        """
        Quick chat method
        """
        return self.process_conversation(user_query, {'session_id': session_id}, user_id)

class PreprocessingOrchestratorWrapper:
    """
    Wrapper cho PreprocessingOrchestrator
    """

    def __init__(self, pool_manager: AgentPoolManager):
        self.pool_manager = pool_manager

    def process_video(self, video_path: str, config=None):
        """
        Sync wrapper cho video processing
        """
        try:
            loop = asyncio.get_event_loop()

            result = loop.run_until_complete(
                self.pool_manager.execute_agent_task(
                    agent_name='preprocessing_orchestrator',
                    method_name='process_video',
                    kwargs={
                        'video_path': video_path,
                        'config': config
                    }
                )
            )

            return result

        except Exception as e:
            return AgentResult(
                status="error",
                result=None,
                error_message=str(e),
                execution_time=0.0,
                agent_name="preprocessing_orchestrator"
            )

#======================================================================================================================================
# MOCK CLASSES
#======================================================================================================================================

class MockConversationOrchestrator:
    """
    Mock conversation orchestrator khi Dev1 chưa implement
    """

    def process_conversation(self, user_query: str, session_context: Dict, user_id: str):
        """
        Mock conversation processing
        """
        mock_result = {
            "final_response": {
                "main_answer": f"Mock response for query: '{user_query}'. This is a placeholder response while waiting for Dev1's actual implementation.",
                "media_references": [],
                "follow_up_suggestions": [
                    "Tell me more about this topic",
                    "Show me related videos",
                    "Explain in more detail"
                ]
            },
            "query_understanding_result": {
                "intent": "general_question",
                "entities": ["mock_entity"],
                "confidence": 0.8
            },
            "retrieval_result": {
                "retrieved_videos": []
            },
            "agent_executions": ["mock_agent"],
            "success_rate": 1.0,
            "status": "completed"
        }

        return AgentResult(
            status="success",
            result=mock_result,
            error_message=None,
            execution_time=1.0,
            agent_name="mock_conversation_orchestrator"
        )

    def quick_chat(self, user_query: str, session_id: str, user_id: str):
        """
        Mock quick chat
        """
        return self.process_conversation(user_query, {'session_id': session_id}, user_id)

class MockPreprocessingOrchestrator:
    """
    Mock preprocessing orchestrator
    """

    def process_video(self, video_path: str, config=None):
        """
        Mock video processing
        """
        mock_result = {
            "pipeline_id": f"mock_pipeline_{int(time.time())}",
            "stages_completed": 5,
            "processing_time": 30.0,
            "success_rate": 1.0,
            "extracted_features": ["mock_feature_1", "mock_feature_2"],
            "analysis_results": {
                "content_type": "educational",
                "main_topics": ["technology", "tutorial"],
                "quality_score": 0.85
            },
            "indexing_completed": True,
            "search_ready": True
        }

        return AgentResult(
            status="success",
            result=mock_result,
            error_message=None,
            execution_time=30.0,
            agent_name="mock_preprocessing_orchestrator"
        )

#======================================================================================================================================
# GLOBAL INSTANCE
#======================================================================================================================================

# Tạo global instance để sử dụng trong toàn bộ API
agents_manager = AgentsManager()

#======================================================================================================================================
# EXPORTS
#======================================================================================================================================

__all__ = [
    # Main classes
    "AgentsManager",
    "AgentPoolManager",

    # Data classes
    "AgentResult",
    "AgentPoolStats",

    # Enums
    "AgentStatus",
    "TaskPriority",

    # Wrapper classes
    "ConversationOrchestratorWrapper",
    "PreprocessingOrchestratorWrapper",

    # Global instance
    "agents_manager"
]

print(f"[{datetime.utcnow()}] Agents Manager module loaded successfully by user: xthanh1910")