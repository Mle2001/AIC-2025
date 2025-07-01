"""
Context Manager Agent - Quản lý conversation context và memory
Conversation Memory, Context Tracking, Session State, User Preferences
Phần của Phase 2: Competition - Real-time context management
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time
from datetime import datetime

from ..base_agent import ConversationalAgent, AgentResponse

# Structured output models
class ConversationTurn(BaseModel):
    """Individual conversation turn"""
    turn_id: str = Field(..., description="Unique turn ID")
    timestamp: str = Field(..., description="Turn timestamp")
    
    # Content
    user_message: str = Field(..., description="User message")
    assistant_response: str = Field(..., description="Assistant response")
    
    # Context
    intent: str = Field(..., description="User intent")
    entities: List[str] = Field(default_factory=list, description="Mentioned entities")
    topics: List[str] = Field(default_factory=list, description="Discussion topics")
    
    # References
    video_references: List[str] = Field(default_factory=list, description="Referenced videos")
    timestamp_references: List[float] = Field(default_factory=list, description="Referenced timestamps")
    
    # Metadata
    satisfaction_score: Optional[float] = Field(None, description="User satisfaction (0-1)")
    resolved: bool = Field(default=False, description="Query resolved successfully")

class UserPreferences(BaseModel):
    """User preferences và settings"""
    user_id: str = Field(..., description="User ID")
    
    # Content preferences
    preferred_content_types: List[str] = Field(default_factory=list, description="Preferred content types")
    favorite_topics: List[str] = Field(default_factory=list, description="Favorite topics")
    language_preference: str = Field(default="en", description="Preferred language")
    
    # Search preferences
    preferred_search_mode: str = Field(default="hybrid", description="Preferred search mode")
    preferred_detail_level: str = Field(default="medium", description="Preferred explanation detail")
    max_results_preference: int = Field(default=10, description="Preferred number of results")
    
    # Interaction preferences
    communication_style: str = Field(default="conversational", description="Preferred communication style")
    feedback_frequency: str = Field(default="normal", description="Feedback frequency preference")
    help_level: str = Field(default="medium", description="Desired help level")
    
    # History
    last_updated: str = Field(..., description="Last preference update")
    update_count: int = Field(default=1, description="Number of updates")

class SessionContext(BaseModel):
    """Current session context"""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    start_time: str = Field(..., description="Session start time")
    
    # Current state
    current_topic: Optional[str] = Field(None, description="Current discussion topic")
    current_video: Optional[str] = Field(None, description="Currently focused video")
    current_timestamp: Optional[float] = Field(None, description="Current timestamp position")
    
    # Context tracking
    active_entities: List[str] = Field(default_factory=list, description="Currently relevant entities")
    mentioned_videos: List[str] = Field(default_factory=list, description="Videos mentioned this session")
    search_history: List[str] = Field(default_factory=list, description="Search queries this session")
    
    # Conversation flow
    conversation_turns: List[ConversationTurn] = Field(default_factory=list, description="Conversation history")
    pending_clarifications: List[str] = Field(default_factory=list, description="Pending clarification questions")
    unresolved_queries: List[str] = Field(default_factory=list, description="Unresolved user queries")
    
    # Preferences (session copy)
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences")

class ContextUpdate(BaseModel):
    """Context update information"""
    update_type: str = Field(..., description="Update type: add_turn, update_topic, add_entity, etc.")
    update_content: Dict[str, Any] = Field(..., description="Update content")
    timestamp: str = Field(..., description="Update timestamp")
    confidence: float = Field(default=1.0, description="Update confidence")

class ContextAnalysis(BaseModel):
    """Context analysis result"""
    analysis_id: str = Field(..., description="Analysis ID")
    session_id: str = Field(..., description="Session ID")
    
    # Current state analysis
    conversation_stage: str = Field(..., description="Conversation stage: beginning, middle, end")
    topic_stability: str = Field(..., description="Topic stability: stable, shifting, unclear")
    user_engagement: str = Field(..., description="User engagement: high, medium, low")
    
    # Trends
    topic_trends: List[str] = Field(default_factory=list, description="Emerging topic trends")
    entity_trends: List[str] = Field(default_factory=list, description="Frequently mentioned entities")
    search_patterns: List[str] = Field(default_factory=list, description="Search behavior patterns")
    
    # Recommendations
    next_actions: List[str] = Field(default_factory=list, description="Recommended next actions")
    clarification_needs: List[str] = Field(default_factory=list, description="Needed clarifications")
    
    # Quality metrics
    context_coherence: float = Field(..., description="Context coherence score (0-1)")
    information_completeness: float = Field(..., description="Information completeness (0-1)")
    
    # Processing info
    analysis_time: float = Field(..., description="Analysis processing time")
    status: str = Field(..., description="Analysis status")

class ContextManagerAgent(ConversationalAgent):
    """
    Agent chuyên quản lý conversation context và memory
    Track conversation state, user preferences, session management
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ContextManager",
            model_type="gpt-4o",
            **kwargs
        )
        
        self.set_instructions([
            "You are a conversation context and memory management specialist.",
            "Your role is to maintain comprehensive conversation context:",
            "- Track conversation history and current state",
            "- Manage user preferences and personalization",
            "- Maintain entity and topic context across turns",
            "- Identify conversation patterns and trends",
            "- Detect context shifts and topic changes",
            "- Provide context-aware recommendations",
            "- Handle ambiguous references and pronouns",
            "Always maintain consistency and continuity in conversations.",
            "Focus on enhancing user experience through smart context management."
        ])
        
        # Context manager doesn't need structured output by default
        # Will use it selectively for specific operations
        
    def update_context(self, session_context: SessionContext, context_update: ContextUpdate, **kwargs) -> AgentResponse:
        """
        Update session context với new information
        
        Args:
            session_context: Current session context
            context_update: New context update
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Create context update prompt
            prompt = self._create_update_prompt(session_context, context_update)
            
            # Process update
            response = self.run_with_timing(prompt, **kwargs)
            
            # Apply update to session context
            updated_context = self._apply_context_update(session_context, context_update)
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="context_update",
                status="success",
                result={
                    "updated_context": updated_context.dict(),
                    "update_applied": True,
                    "update_type": context_update.update_type
                },
                execution_time=execution_time,
                metadata={
                    "session_id": session_context.session_id,
                    "update_type": context_update.update_type,
                    "context_size": len(session_context.conversation_turns)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Context update failed: {str(e)}")
            
            return self._create_response(
                task_type="context_update",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_update_prompt(self, session_context: SessionContext, update: ContextUpdate) -> str:
        """Tạo prompt cho context update"""
        
        prompt = f"""
        Update conversation context for session: {session_context.session_id}
        
        CURRENT CONTEXT STATE:
        - Session Duration: {len(session_context.conversation_turns)} turns
        - Current Topic: {session_context.current_topic or 'None'}
        - Current Video: {session_context.current_video or 'None'}
        - Active Entities: {session_context.active_entities}
        - Mentioned Videos: {session_context.mentioned_videos}
        - Recent Searches: {session_context.search_history[-3:] if session_context.search_history else []}
        
        UPDATE TO APPLY:
        - Type: {update.update_type}
        - Content: {update.update_content}
        - Timestamp: {update.timestamp}
        - Confidence: {update.confidence}
        
        CONTEXT UPDATE TASKS:
        1. Analyze the impact of this update on current context
        2. Identify what context elements need to change
        3. Determine if topic or focus has shifted
        4. Update entity relevance and mentions
        5. Track conversation progression and patterns
        6. Identify any clarification needs
        
        Provide analysis of how this update affects the conversation context.
        """
        
        return prompt
    
    def _apply_context_update(self, context: SessionContext, update: ContextUpdate) -> SessionContext:
        """Apply context update to session context"""
        
        update_type = update.update_type
        content = update.update_content
        
        if update_type == "add_conversation_turn":
            # Add new conversation turn
            new_turn = ConversationTurn(
                turn_id=content.get("turn_id", f"turn_{len(context.conversation_turns)}"),
                timestamp=update.timestamp,
                user_message=content.get("user_message", ""),
                assistant_response=content.get("assistant_response", ""),
                intent=content.get("intent", "unknown"),
                entities=content.get("entities", []),
                topics=content.get("topics", []),
                video_references=content.get("video_references", []),
                timestamp_references=content.get("timestamp_references", [])
            )
            context.conversation_turns.append(new_turn)
            
            # Update active entities
            for entity in new_turn.entities:
                if entity not in context.active_entities:
                    context.active_entities.append(entity)
            
            # Update mentioned videos
            for video in new_turn.video_references:
                if video not in context.mentioned_videos:
                    context.mentioned_videos.append(video)
        
        elif update_type == "update_current_topic":
            context.current_topic = content.get("topic")
        
        elif update_type == "update_current_video":
            context.current_video = content.get("video_id")
            context.current_timestamp = content.get("timestamp")
        
        elif update_type == "add_search_query":
            search_query = content.get("query", "")
            context.search_history.append(search_query)
            # Keep only last 20 searches
            if len(context.search_history) > 20:
                context.search_history = context.search_history[-20:]
        
        elif update_type == "add_clarification":
            clarification = content.get("clarification", "")
            context.pending_clarifications.append(clarification)
        
        elif update_type == "resolve_query":
            query = content.get("query", "")
            if query in context.unresolved_queries:
                context.unresolved_queries.remove(query)
        
        return context
    
    def analyze_context(self, session_context: SessionContext, **kwargs) -> AgentResponse:
        """
        Analyze current conversation context
        
        Args:
            session_context: Current session context
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Set response model for analysis
            self.agent.response_model = ContextAnalysis
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(session_context)
            
            # Run analysis
            response = self.agent.run(prompt, user_id=kwargs.get('user_id'))
            
            # Validate response
            if not isinstance(response, ContextAnalysis):
                result = self._parse_analysis_response(response.content, session_context)
            else:
                result = response
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="context_analysis",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "session_id": session_context.session_id,
                    "conversation_length": len(session_context.conversation_turns),
                    "context_coherence": result.context_coherence,
                    "completeness": result.information_completeness
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Context analysis failed: {str(e)}")
            
            return self._create_response(
                task_type="context_analysis",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
        
        finally:
            # Reset response model
            self.agent.response_model = None
    
    def _create_analysis_prompt(self, context: SessionContext) -> str:
        """Tạo prompt cho context analysis"""
        
        # Calculate session stats
        turn_count = len(context.conversation_turns)
        recent_turns = context.conversation_turns[-5:] if context.conversation_turns else []
        
        prompt = f"""
        Analyze conversation context for session: {context.session_id}
        
        SESSION OVERVIEW:
        - User: {context.user_id}
        - Duration: {turn_count} conversation turns
        - Start Time: {context.start_time}
        - Current Topic: {context.current_topic or 'Not established'}
        - Current Video: {context.current_video or 'None focused'}
        
        CONTEXT STATE:
        - Active Entities: {context.active_entities}
        - Mentioned Videos: {context.mentioned_videos}
        - Search History: {context.search_history[-5:] if context.search_history else []}
        - Pending Clarifications: {context.pending_clarifications}
        - Unresolved Queries: {context.unresolved_queries}
        
        RECENT CONVERSATION:
        """
        
        for i, turn in enumerate(recent_turns):
            prompt += f"""
        Turn {i+1} ({turn.timestamp}):
        - User: {turn.user_message[:100]}...
        - Intent: {turn.intent}
        - Entities: {turn.entities}
        - Topics: {turn.topics}
        """
        
        prompt += """
        
        ANALYSIS TASKS:
        
        1. CONVERSATION STAGE ASSESSMENT:
           - Determine if conversation is beginning, middle, or ending
           - Assess conversation flow and progression
           - Identify conversation patterns
        
        2. TOPIC STABILITY ANALYSIS:
           - Analyze topic consistency across turns
           - Identify topic shifts and changes
           - Assess topic coherence and focus
        
        3. USER ENGAGEMENT EVALUATION:
           - Assess user engagement level based on message patterns
           - Identify engagement trends (increasing/decreasing)
           - Evaluate query complexity and specificity
        
        4. TREND IDENTIFICATION:
           - Identify emerging topic trends
           - Track frequently mentioned entities
           - Analyze search behavior patterns
        
        5. RECOMMENDATION GENERATION:
           - Suggest next actions for better conversation flow
           - Identify needed clarifications
           - Recommend context improvements
        
        6. QUALITY METRICS:
           - Calculate context coherence score (0-1)
           - Assess information completeness (0-1)
           - Evaluate overall context quality
        
        OUTPUT REQUIREMENTS:
        - Return complete ContextAnalysis object
        - Provide detailed analysis of all aspects
        - Include actionable recommendations
        - Calculate meaningful quality metrics
        """
        
        return prompt
    
    def _parse_analysis_response(self, response_content: str, context: SessionContext) -> ContextAnalysis:
        """Fallback parsing nếu structured output fails"""
        return ContextAnalysis(
            analysis_id=f"analysis_{int(time.time())}",
            session_id=context.session_id,
            conversation_stage="middle",
            topic_stability="stable",
            user_engagement="medium",
            topic_trends=[],
            entity_trends=[],
            search_patterns=[],
            next_actions=["continue_conversation"],
            clarification_needs=[],
            context_coherence=0.7,
            information_completeness=0.6,
            analysis_time=0,
            status="parsed_fallback"
        )
    
    def resolve_references(self, current_message: str, session_context: SessionContext, **kwargs) -> AgentResponse:
        """
        Resolve ambiguous references trong current message
        
        Args:
            current_message: Current user message
            session_context: Session context for reference resolution
            **kwargs: Additional parameters
        """
        prompt = f"""
        Resolve ambiguous references in this message: "{current_message}"
        
        CONVERSATION CONTEXT:
        - Current Topic: {session_context.current_topic}
        - Current Video: {session_context.current_video}
        - Current Timestamp: {session_context.current_timestamp}
        - Active Entities: {session_context.active_entities}
        - Mentioned Videos: {session_context.mentioned_videos}
        
        RECENT CONVERSATION:
        """
        
        # Add last 3 turns for context
        for turn in session_context.conversation_turns[-3:]:
            prompt += f"\nUser: {turn.user_message}\nAssistant: {turn.assistant_response[:100]}...\n"
        
        prompt += """
        
        REFERENCE RESOLUTION TASKS:
        1. Identify pronouns and ambiguous references (it, that, this, there, etc.)
        2. Map references to specific entities from context
        3. Resolve temporal references (now, then, earlier, etc.)
        4. Clarify any remaining ambiguities
        5. Provide resolved message with clear references
        
        Return the message with resolved references and explanation of resolutions made.
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def manage_user_preferences(self, user_id: str, preference_updates: Dict[str, Any], **kwargs) -> AgentResponse:
        """
        Update và manage user preferences
        
        Args:
            user_id: User ID
            preference_updates: Preference updates to apply
            **kwargs: Additional parameters
        """
        # Load current preferences (mock implementation)
        current_preferences = UserPreferences(
            user_id=user_id,
            last_updated=datetime.now().isoformat()
        )
        
        prompt = f"""
        Update user preferences for user: {user_id}
        
        CURRENT PREFERENCES:
        - Content Types: {current_preferences.preferred_content_types}
        - Topics: {current_preferences.favorite_topics}
        - Search Mode: {current_preferences.preferred_search_mode}
        - Detail Level: {current_preferences.preferred_detail_level}
        - Communication Style: {current_preferences.communication_style}
        
        PROPOSED UPDATES:
        {preference_updates}
        
        TASKS:
        1. Validate preference updates
        2. Apply updates to current preferences
        3. Identify any conflicts or issues
        4. Generate updated preference profile
        5. Provide personalization recommendations
        
        Return updated preferences and recommendations for personalization.
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def detect_context_shift(self, session_context: SessionContext, new_message: str, **kwargs) -> AgentResponse:
        """
        Detect significant context shifts trong conversation
        
        Args:
            session_context: Current session context
            new_message: New user message
            **kwargs: Additional parameters
        """
        prompt = f"""
        Detect context shift in conversation for new message: "{new_message}"
        
        CURRENT CONTEXT:
        - Topic: {session_context.current_topic}
        - Video Focus: {session_context.current_video}
        - Active Entities: {session_context.active_entities}
        - Recent Searches: {session_context.search_history[-3:] if session_context.search_history else []}
        
        CONTEXT SHIFT DETECTION:
        1. Compare new message topic with current topic
        2. Identify new entities or concepts
        3. Detect changes in user intent or focus
        4. Assess if this represents a major shift
        5. Determine if context reset is needed
        
        Return shift detection results:
        - Shift detected: yes/no
        - Shift type: topic, entity, intent, video, complete
        - Shift magnitude: minor, moderate, major
        - Recommended actions: continue, update, reset
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def get_context_summary(self, session_context: SessionContext, **kwargs) -> AgentResponse:
        """
        Generate summary của current context state
        
        Args:
            session_context: Session context to summarize
            **kwargs: Additional parameters
        """
        prompt = f"""
        Generate comprehensive summary of conversation context:
        
        SESSION INFO:
        - Session ID: {session_context.session_id}
        - User: {session_context.user_id}
        - Duration: {len(session_context.conversation_turns)} turns
        - Start: {session_context.start_time}
        
        CURRENT STATE:
        - Topic: {session_context.current_topic}
        - Video: {session_context.current_video}
        - Timestamp: {session_context.current_timestamp}
        - Active Entities: {session_context.active_entities}
        
        CONVERSATION ACTIVITY:
        - Videos Discussed: {session_context.mentioned_videos}
        - Search Queries: {len(session_context.search_history)}
        - Pending Items: {len(session_context.pending_clarifications + session_context.unresolved_queries)}
        
        SUMMARY REQUIREMENTS:
        1. Summarize main conversation topics
        2. Highlight key entities and videos discussed
        3. Note user interests and preferences
        4. Identify unresolved items
        5. Provide context continuity information
        
        Generate clear, concise context summary for handoff or reference.
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def get_context_management_stats(self, session_contexts: List[SessionContext]) -> Dict[str, Any]:
        """
        Generate statistics về context management performance
        
        Args:
            session_contexts: List of session contexts to analyze
        """
        if not session_contexts:
            return {"error": "No session contexts to analyze"}
        
        total_sessions = len(session_contexts)
        total_turns = sum(len(ctx.conversation_turns) for ctx in session_contexts)
        
        # Session duration distribution
        session_lengths = [len(ctx.conversation_turns) for ctx in session_contexts]
        avg_session_length = sum(session_lengths) / len(session_lengths)
        
        # Topic stability analysis
        sessions_with_stable_topics = len([
            ctx for ctx in session_contexts 
            if ctx.current_topic and len(ctx.conversation_turns) > 2
        ])
        
        # User engagement patterns
        active_sessions = len([
            ctx for ctx in session_contexts 
            if len(ctx.conversation_turns) >= 5
        ])
        
        return {
            "total_sessions": total_sessions,
            "total_conversation_turns": total_turns,
            "avg_session_length": round(avg_session_length, 1),
            "topic_stability_rate": round(sessions_with_stable_topics / total_sessions, 2),
            "user_engagement_rate": round(active_sessions / total_sessions, 2),
            "session_length_distribution": {
                "short": len([l for l in session_lengths if l < 3]),
                "medium": len([l for l in session_lengths if 3 <= l < 10]),
                "long": len([l for l in session_lengths if l >= 10])
            },
            "context_management_quality": {
                "entity_tracking": "active",
                "reference_resolution": "enabled",
                "preference_adaptation": "enabled",
                "conversation_continuity": "maintained"
            }
        }