"""
Conversation Orchestrator - Điều phối conversation flow trong Phase 2
Coordinate các conversational agents để handle user queries trong competition
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time
import asyncio
from enum import Enum

from ..base_agent import ConversationalAgent, AgentResponse
from ..conversational.query_understanding_agent import QueryUnderstandingAgent
from ..conversational.retrieval_agent import VideoRetrievalAgent  
from ..conversational.explanation_agent import ContentExplainerAgent
from ..conversational.context_manager_agent import ContextManagerAgent, SessionContext, ContextUpdate
from ..conversational.response_synthesis_agent import ResponseSynthesisAgent, SynthesisRequest

# Structured output models
class ConversationState(str, Enum):
    """Conversation states"""
    INITIALIZING = "initializing"
    UNDERSTANDING = "understanding" 
    RETRIEVING = "retrieving"
    EXPLAINING = "explaining"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"

class AgentExecution(BaseModel):
    """Individual agent execution result"""
    agent_name: str = Field(..., description="Agent name")
    execution_time: float = Field(..., description="Execution time")
    status: str = Field(..., description="Execution status")
    result: Dict[str, Any] = Field(default_factory=dict, description="Agent result")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class ConversationFlow(BaseModel):
    """Complete conversation flow result"""
    flow_id: str = Field(..., description="Unique flow ID")
    user_query: str = Field(..., description="Original user query")
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    
    # Flow execution
    state: ConversationState = Field(..., description="Current conversation state")
    agent_executions: List[AgentExecution] = Field(default_factory=list, description="Agent execution results")
    
    # Results from each agent
    query_understanding: Optional[Dict[str, Any]] = Field(None, description="Query understanding results")
    retrieval_results: Optional[Dict[str, Any]] = Field(None, description="Video retrieval results")
    explanation_results: Optional[Dict[str, Any]] = Field(None, description="Content explanation results")
    context_updates: Optional[Dict[str, Any]] = Field(None, description="Context manager updates")
    final_response: Optional[Dict[str, Any]] = Field(None, description="Synthesized final response")
    
    # Flow metadata
    total_execution_time: float = Field(..., description="Total flow execution time")
    success_rate: float = Field(..., description="Success rate of agent executions")
    user_satisfaction_score: Optional[float] = Field(None, description="User satisfaction score")
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Status
    status: str = Field(..., description="Overall flow status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class ConversationConfig(BaseModel):
    """Configuration cho conversation orchestration"""
    # Agent enables
    enable_query_understanding: bool = Field(default=True, description="Enable query understanding")
    enable_retrieval: bool = Field(default=True, description="Enable video retrieval")
    enable_explanation: bool = Field(default=True, description="Enable content explanation")
    enable_context_management: bool = Field(default=True, description="Enable context management")
    enable_response_synthesis: bool = Field(default=True, description="Enable response synthesis")
    
    # Execution strategy
    execution_mode: str = Field(default="sequential", description="Execution mode: sequential, parallel, adaptive")
    timeout_per_agent: int = Field(default=30, description="Timeout per agent (seconds)")
    max_retries: int = Field(default=2, description="Max retries per agent")
    
    # Quality thresholds
    min_understanding_confidence: float = Field(default=0.6, description="Min query understanding confidence")
    min_retrieval_relevance: float = Field(default=0.3, description="Min retrieval relevance")
    min_explanation_quality: float = Field(default=0.5, description="Min explanation quality")
    
    # User experience
    provide_intermediate_feedback: bool = Field(default=True, description="Provide intermediate feedback")
    stream_response: bool = Field(default=True, description="Stream response as available")
    max_response_time: int = Field(default=10, description="Max total response time (seconds)")

class ConversationOrchestrator(ConversationalAgent):
    """
    Orchestrator điều phối conversation flow trong competition phase
    Coordinate tất cả conversational agents để xử lý user queries
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ConversationOrchestrator",
            model_type="gpt-4o",
            **kwargs
        )
        
        # Initialize agents
        self.query_understanding_agent = QueryUnderstandingAgent(**kwargs)
        self.retrieval_agent = VideoRetrievalAgent(**kwargs)
        self.explanation_agent = ContentExplainerAgent(**kwargs)
        self.context_manager = ContextManagerAgent(**kwargs)
        self.response_synthesis_agent = ResponseSynthesisAgent(**kwargs)
        
        self.set_instructions([
            "You are the conversation orchestration coordinator.",
            "Your role is to coordinate multiple agents to handle user queries:",
            "- Orchestrate query understanding, retrieval, explanation, and synthesis",
            "- Manage conversation context and user preferences",
            "- Ensure optimal performance and user experience",
            "- Handle errors gracefully and provide fallback responses",
            "- Adapt strategy based on query characteristics and user needs",
            "Focus on delivering fast, accurate, and satisfying responses."
        ])
        
        self.agent.response_model = ConversationFlow
        
    def process_conversation(self, 
                           user_query: str,
                           session_context: SessionContext,
                           config: Optional[ConversationConfig] = None,
                           **kwargs) -> AgentResponse:
        """
        Process complete conversation flow
        
        Args:
            user_query: User's query
            session_context: Current session context
            config: Conversation configuration
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Use default config if not provided
            if config is None:
                config = ConversationConfig()
            
            # Initialize conversation flow
            flow = ConversationFlow(
                flow_id=f"flow_{int(time.time())}",
                user_query=user_query,
                session_id=session_context.session_id,
                user_id=session_context.user_id,
                state=ConversationState.INITIALIZING,
                agent_executions=[],
                total_execution_time=0,
                success_rate=0,
                performance_metrics={},
                status="processing"
            )
            
            # Execute conversation flow
            if config.execution_mode == "sequential":
                flow = self._execute_sequential_flow(flow, user_query, session_context, config, **kwargs)
            elif config.execution_mode == "parallel":
                flow = self._execute_parallel_flow(flow, user_query, session_context, config, **kwargs)
            else:  # adaptive
                flow = self._execute_adaptive_flow(flow, user_query, session_context, config, **kwargs)
            
            # Calculate final metrics
            flow.total_execution_time = time.time() - start_time
            flow.success_rate = self._calculate_success_rate(flow)
            flow.performance_metrics = self._calculate_performance_metrics(flow)
            
            # Update conversation state
            if flow.final_response:
                flow.state = ConversationState.COMPLETED
                flow.status = "success"
            else:
                flow.state = ConversationState.ERROR
                flow.status = "partial_success"
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="conversation_orchestration",
                status="success" if flow.status == "success" else "partial",
                result=flow.dict(),
                execution_time=execution_time,
                metadata={
                    "query": user_query,
                    "agents_executed": len(flow.agent_executions),
                    "success_rate": flow.success_rate,
                    "total_time": flow.total_execution_time,
                    "final_response_available": flow.final_response is not None
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Conversation orchestration failed: {str(e)}")
            
            return self._create_response(
                task_type="conversation_orchestration",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _execute_sequential_flow(self, 
                                flow: ConversationFlow,
                                user_query: str,
                                session_context: SessionContext,
                                config: ConversationConfig,
                                **kwargs) -> ConversationFlow:
        """Execute agents sequentially"""
        
        # Step 1: Query Understanding
        if config.enable_query_understanding:
            flow.state = ConversationState.UNDERSTANDING
            understanding_result = self._execute_agent_with_timeout(
                self.query_understanding_agent,
                "process",
                {"user_query": user_query, "context": session_context.dict()},
                config.timeout_per_agent,
                **kwargs
            )
            flow.agent_executions.append(understanding_result)
            
            if understanding_result.status == "success":
                flow.query_understanding = understanding_result.result
        
        # Step 2: Video Retrieval (if understanding successful)
        if config.enable_retrieval and flow.query_understanding:
            flow.state = ConversationState.RETRIEVING
            
            # Check understanding confidence
            understanding_confidence = flow.query_understanding.get("confidence_score", 0)
            if understanding_confidence >= config.min_understanding_confidence:
                
                retrieval_result = self._execute_agent_with_timeout(
                    self.retrieval_agent,
                    "process",
                    {"query_understanding": flow.query_understanding},
                    config.timeout_per_agent,
                    **kwargs
                )
                flow.agent_executions.append(retrieval_result)
                
                if retrieval_result.status == "success":
                    flow.retrieval_results = retrieval_result.result
        
        # Step 3: Content Explanation (if retrieval successful and needed)
        if config.enable_explanation and flow.retrieval_results:
            flow.state = ConversationState.EXPLAINING
            
            # Check if explanation is needed based on intent
            intent_type = flow.query_understanding.get("intent", {}).get("intent_type", "search")
            if intent_type in ["explain", "analyze", "describe"]:
                
                # Create explanation request
                explanation_request = self._create_explanation_request(
                    user_query, 
                    flow.query_understanding,
                    flow.retrieval_results
                )
                
                explanation_result = self._execute_agent_with_timeout(
                    self.explanation_agent,
                    "process",
                    explanation_request,
                    config.timeout_per_agent,
                    **kwargs
                )
                flow.agent_executions.append(explanation_result)
                
                if explanation_result.status == "success":
                    flow.explanation_results = explanation_result.result
        
        # Step 4: Context Update
        if config.enable_context_management:
            context_update = ContextUpdate(
                update_type="add_conversation_turn",
                update_content={
                    "user_message": user_query,
                    "intent": flow.query_understanding.get("intent", {}).get("intent_type", "unknown") if flow.query_understanding else "unknown",
                    "entities": [e.get("entity_text", "") for e in flow.query_understanding.get("entities", [])] if flow.query_understanding else [],
                    "video_references": [r.get("video_id", "") for r in flow.retrieval_results.get("results", [])] if flow.retrieval_results else []
                },
                timestamp=str(time.time())
            )
            
            context_result = self._execute_agent_with_timeout(
                self.context_manager,
                "update_context",
                {"session_context": session_context, "context_update": context_update},
                config.timeout_per_agent,
                **kwargs
            )
            flow.agent_executions.append(context_result)
            
            if context_result.status == "success":
                flow.context_updates = context_result.result
        
        # Step 5: Response Synthesis
        if config.enable_response_synthesis:
            flow.state = ConversationState.SYNTHESIZING
            
            synthesis_request = SynthesisRequest(
                query=user_query,
                query_understanding=flow.query_understanding or {},
                retrieval_results=flow.retrieval_results.get("results", []) if flow.retrieval_results else [],
                explanations=[flow.explanation_results] if flow.explanation_results else [],
                context_info=session_context.dict(),
                user_preferences=session_context.user_preferences.dict() if session_context.user_preferences else {}
            )
            
            synthesis_result = self._execute_agent_with_timeout(
                self.response_synthesis_agent,
                "process",
                {"synthesis_request": synthesis_request},
                config.timeout_per_agent,
                **kwargs
            )
            flow.agent_executions.append(synthesis_result)
            
            if synthesis_result.status == "success":
                flow.final_response = synthesis_result.result
        
        return flow
    
    def _execute_parallel_flow(self, 
                              flow: ConversationFlow,
                              user_query: str,
                              session_context: SessionContext,
                              config: ConversationConfig,
                              **kwargs) -> ConversationFlow:
        """Execute compatible agents in parallel"""
        
        # Step 1: Query Understanding (must be first)
        flow.state = ConversationState.UNDERSTANDING
        understanding_result = self._execute_agent_with_timeout(
            self.query_understanding_agent,
            "process",
            {"user_query": user_query, "context": session_context.dict()},
            config.timeout_per_agent,
            **kwargs
        )
        flow.agent_executions.append(understanding_result)
        
        if understanding_result.status == "success":
            flow.query_understanding = understanding_result.result
            
            # Step 2: Parallel execution of retrieval and context update
            parallel_tasks = []
            
            if config.enable_retrieval:
                parallel_tasks.append(("retrieval", self.retrieval_agent, "process", 
                                     {"query_understanding": flow.query_understanding}))
            
            if config.enable_context_management:
                context_update = ContextUpdate(
                    update_type="add_conversation_turn",
                    update_content={"user_message": user_query},
                    timestamp=str(time.time())
                )
                parallel_tasks.append(("context", self.context_manager, "update_context",
                                     {"session_context": session_context, "context_update": context_update}))
            
            # Execute parallel tasks
            parallel_results = self._execute_parallel_agents(parallel_tasks, config.timeout_per_agent, **kwargs)
            
            for task_name, result in parallel_results:
                flow.agent_executions.append(result)
                if task_name == "retrieval" and result.status == "success":
                    flow.retrieval_results = result.result
                elif task_name == "context" and result.status == "success":
                    flow.context_updates = result.result
            
            # Step 3: Response Synthesis (depends on previous results)
            if config.enable_response_synthesis:
                flow.state = ConversationState.SYNTHESIZING
                
                synthesis_request = SynthesisRequest(
                    query=user_query,
                    query_understanding=flow.query_understanding,
                    retrieval_results=flow.retrieval_results.get("results", []) if flow.retrieval_results else [],
                    explanations=[],
                    context_info=session_context.dict()
                )
                
                synthesis_result = self._execute_agent_with_timeout(
                    self.response_synthesis_agent,
                    "process",
                    {"synthesis_request": synthesis_request},
                    config.timeout_per_agent,
                    **kwargs
                )
                flow.agent_executions.append(synthesis_result)
                
                if synthesis_result.status == "success":
                    flow.final_response = synthesis_result.result
        
        return flow
    
    def _execute_adaptive_flow(self, 
                              flow: ConversationFlow,
                              user_query: str,
                              session_context: SessionContext,
                              config: ConversationConfig,
                              **kwargs) -> ConversationFlow:
        """Execute agents adaptively based on query characteristics"""
        
        # Always start with query understanding
        flow.state = ConversationState.UNDERSTANDING
        understanding_result = self._execute_agent_with_timeout(
            self.query_understanding_agent,
            "process",
            {"user_query": user_query, "context": session_context.dict()},
            config.timeout_per_agent,
            **kwargs
        )
        flow.agent_executions.append(understanding_result)
        
        if understanding_result.status == "success":
            flow.query_understanding = understanding_result.result
            
            # Adapt based on query characteristics
            intent_type = flow.query_understanding.get("intent", {}).get("intent_type", "search")
            query_complexity = flow.query_understanding.get("query_complexity", "medium")
            
            # Simple search queries - minimal processing
            if intent_type == "search" and query_complexity == "simple":
                flow = self._execute_simple_search_flow(flow, user_query, session_context, config, **kwargs)
            
            # Explanation queries - focus on explanation
            elif intent_type in ["explain", "analyze", "describe"]:
                flow = self._execute_explanation_focused_flow(flow, user_query, session_context, config, **kwargs)
            
            # Complex queries - full processing
            else:
                flow = self._execute_sequential_flow(flow, user_query, session_context, config, **kwargs)
        
        return flow
    
    def _execute_simple_search_flow(self, flow: ConversationFlow, user_query: str, session_context: SessionContext, config: ConversationConfig, **kwargs) -> ConversationFlow:
        """Optimized flow for simple search queries"""
        
        # Just retrieval and synthesis
        if config.enable_retrieval:
            flow.state = ConversationState.RETRIEVING
            retrieval_result = self._execute_agent_with_timeout(
                self.retrieval_agent,
                "search_by_text_only",
                {"text_query": user_query, "max_results": 5},
                config.timeout_per_agent,
                **kwargs
            )
            flow.agent_executions.append(retrieval_result)
            
            if retrieval_result.status == "success":
                flow.retrieval_results = retrieval_result.result
                
                # Quick synthesis
                if config.enable_response_synthesis and flow.retrieval_results.get("results"):
                    top_result = flow.retrieval_results["results"][0]
                    synthesis_result = self._execute_agent_with_timeout(
                        self.response_synthesis_agent,
                        "synthesize_simple_answer",
                        {"query": user_query, "top_result": top_result},
                        config.timeout_per_agent,
                        **kwargs
                    )
                    flow.agent_executions.append(synthesis_result)
                    
                    if synthesis_result.status == "success":
                        flow.final_response = synthesis_result.result
        
        return flow
    
    def _execute_explanation_focused_flow(self, flow: ConversationFlow, user_query: str, session_context: SessionContext, config: ConversationConfig, **kwargs) -> ConversationFlow:
        """Flow optimized for explanation queries"""
        
        # Retrieval first, then explanation, then synthesis
        if config.enable_retrieval:
            retrieval_result = self._execute_agent_with_timeout(
                self.retrieval_agent,
                "process",
                {"query_understanding": flow.query_understanding},
                config.timeout_per_agent,
                **kwargs
            )
            flow.agent_executions.append(retrieval_result)
            
            if retrieval_result.status == "success":
                flow.retrieval_results = retrieval_result.result
                
                # Focus on explanation
                if config.enable_explanation:
                    explanation_request = self._create_explanation_request(
                        user_query,
                        flow.query_understanding,
                        flow.retrieval_results
                    )
                    
                    explanation_result = self._execute_agent_with_timeout(
                        self.explanation_agent,
                        "process",
                        explanation_request,
                        config.timeout_per_agent,
                        **kwargs
                    )
                    flow.agent_executions.append(explanation_result)
                    
                    if explanation_result.status == "success":
                        flow.explanation_results = explanation_result.result
                        
                        # Synthesis focused on explanation
                        if config.enable_response_synthesis:
                            synthesis_result = self._execute_agent_with_timeout(
                                self.response_synthesis_agent,
                                "synthesize_explanation_response",
                                {"query": user_query, "explanation_data": flow.explanation_results},
                                config.timeout_per_agent,
                                **kwargs
                            )
                            flow.agent_executions.append(synthesis_result)
                            
                            if synthesis_result.status == "success":
                                flow.final_response = synthesis_result.result
        
        return flow
    
    def _execute_agent_with_timeout(self, agent, method_name: str, args: Dict[str, Any], timeout: int, **kwargs) -> AgentExecution:
        """Execute agent method với timeout"""
        start_time = time.time()
        
        try:
            method = getattr(agent, method_name)
            result = method(**args, **kwargs)
            
            execution_time = time.time() - start_time
            
            return AgentExecution(
                agent_name=agent.name,
                execution_time=execution_time,
                status=result.status,
                result=result.result,
                error_message=result.error_message
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return AgentExecution(
                agent_name=agent.name,
                execution_time=execution_time,
                status="error",
                result={},
                error_message=str(e)
            )
    
    def _execute_parallel_agents(self, tasks: List[tuple], timeout: int, **kwargs) -> List[tuple]:
        """Execute multiple agents in parallel"""
        results = []
        
        # Simple parallel execution (trong thực tế sẽ dùng asyncio)
        for task_name, agent, method_name, args in tasks:
            result = self._execute_agent_with_timeout(agent, method_name, args, timeout, **kwargs)
            results.append((task_name, result))
        
        return results
    
    def _create_explanation_request(self, user_query: str, query_understanding: Dict, retrieval_results: Dict) -> Dict[str, Any]:
        """Create explanation request"""
        from ..conversational.explanation_agent import ExplanationRequest
        
        # Determine explanation type based on intent
        intent_type = query_understanding.get("intent", {}).get("intent_type", "explain")
        explanation_type = "scene" if "scene" in user_query.lower() else "full"
        
        # Get target entities
        entities = [e.get("entity_text", "") for e in query_understanding.get("entities", [])]
        
        request = ExplanationRequest(
            request_type=explanation_type,
            video_id=retrieval_results.get("results", [{}])[0].get("video_id", "unknown"),
            entity_names=entities,
            detail_level="medium"
        )
        
        # Mock content data
        content_data = {
            "visual_features": [],
            "audio_features": [],
            "scenes": [],
            "entities": []
        }
        
        return {
            "explanation_request": request,
            "content_data": content_data
        }
    
    def _calculate_success_rate(self, flow: ConversationFlow) -> float:
        """Calculate agent execution success rate"""
        if not flow.agent_executions:
            return 0.0
        
        successful = len([e for e in flow.agent_executions if e.status == "success"])
        return successful / len(flow.agent_executions)
    
    def _calculate_performance_metrics(self, flow: ConversationFlow) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "total_agents": len(flow.agent_executions),
            "successful_agents": len([e for e in flow.agent_executions if e.status == "success"]),
            "avg_agent_time": sum(e.execution_time for e in flow.agent_executions) / max(len(flow.agent_executions), 1),
            "slowest_agent": max((e.execution_time for e in flow.agent_executions), default=0),
            "fastest_agent": min((e.execution_time for e in flow.agent_executions), default=0),
            "response_completeness": 1.0 if flow.final_response else 0.5,
            "context_maintained": 1.0 if flow.context_updates else 0.0
        }
    
    def quick_chat(self, user_query: str, session_id: str, user_id: str, **kwargs) -> AgentResponse:
        """Quick chat interface cho simple queries"""
        
        # Create minimal session context
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=str(time.time()),
            conversation_turns=[],
            active_entities=[],
            mentioned_videos=[],
            search_history=[]
        )
        
        # Use simple config
        config = ConversationConfig(
            execution_mode="adaptive",
            timeout_per_agent=10,
            enable_explanation=False  # Skip explanation for quick chat
        )
        
        return self.process_conversation(user_query, session_context, config, **kwargs)
    
    def get_orchestration_stats(self, flows: List[ConversationFlow]) -> Dict[str, Any]:
        """Analyze orchestration performance"""
        if not flows:
            return {"error": "No flows to analyze"}
        
        total_flows = len(flows)
        successful_flows = len([f for f in flows if f.status == "success"])
        
        # Performance metrics
        execution_times = [f.total_execution_time for f in flows]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        success_rates = [f.success_rate for f in flows]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        return {
            "total_conversations": total_flows,
            "successful_conversations": successful_flows,
            "success_rate": successful_flows / total_flows,
            "avg_execution_time": round(avg_execution_time, 2),
            "avg_agent_success_rate": round(avg_success_rate, 2),
            "performance_distribution": {
                "fast": len([t for t in execution_times if t < 5]),
                "medium": len([t for t in execution_times if 5 <= t < 15]),
                "slow": len([t for t in execution_times if t >= 15])
            },
            "agent_usage": {
                "query_understanding": len([f for f in flows if f.query_understanding]),
                "retrieval": len([f for f in flows if f.retrieval_results]),
                "explanation": len([f for f in flows if f.explanation_results]),
                "synthesis": len([f for f in flows if f.final_response])
            }
        }