"""
Query Understanding Agent - Phân tích và hiểu user queries
Intent Classification, Entity Extraction, Query Expansion, Ambiguity Resolution
Phần của Phase 2: Competition - Real-time conversation
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time

from ..base_agent import ConversationalAgent, AgentResponse

# Structured output models
class Intent(BaseModel):
    """Phân loại intent của user query"""
    intent_type: str = Field(..., description="Main intent: search, explain, compare, list, navigate, etc.")
    sub_intent: str = Field(..., description="Sub-intent: specific_search, general_browse, clarification, etc.")
    confidence: float = Field(..., description="Intent classification confidence (0-1)")
    description: str = Field(..., description="Human-readable intent description")

class Entity(BaseModel):
    """Named entity extracted từ query"""
    entity_text: str = Field(..., description="Original entity text in query")
    entity_type: str = Field(..., description="Entity type: person, object, location, action, time, etc.")
    normalized_form: str = Field(..., description="Normalized entity form")
    aliases: List[str] = Field(default_factory=list, description="Known aliases")
    confidence: float = Field(..., description="Entity extraction confidence")
    start_pos: int = Field(..., description="Start position in query")
    end_pos: int = Field(..., description="End position in query")

class QueryExpansion(BaseModel):
    """Query expansion suggestions"""
    original_query: str = Field(..., description="Original user query")
    expanded_queries: List[str] = Field(default_factory=list, description="Expanded query variations")
    synonyms: List[str] = Field(default_factory=list, description="Synonym suggestions")
    related_terms: List[str] = Field(default_factory=list, description="Related search terms")
    search_strategies: List[str] = Field(default_factory=list, description="Recommended search strategies")

class TemporalInfo(BaseModel):
    """Temporal information in query"""
    has_temporal: bool = Field(default=False, description="Query contains temporal references")
    temporal_type: str = Field(default="none", description="Temporal type: absolute, relative, duration, etc.")
    temporal_expressions: List[str] = Field(default_factory=list, description="Temporal expressions found")
    time_constraints: Dict[str, Any] = Field(default_factory=dict, description="Parsed time constraints")

class ModalityPreference(BaseModel):
    """User's preferred modalities for search"""
    preferred_modalities: List[str] = Field(default_factory=list, description="video, audio, text, image")
    modality_weights: Dict[str, float] = Field(default_factory=dict, description="Relative importance weights")
    explicit_modality: Optional[str] = Field(None, description="Explicitly requested modality")

class QueryUnderstanding(BaseModel):
    """Comprehensive query understanding result"""
    query_id: str = Field(..., description="Unique query ID")
    original_query: str = Field(..., description="Original user query")
    processed_query: str = Field(..., description="Cleaned and processed query")
    
    # Core understanding components
    intent: Intent = Field(..., description="Query intent analysis")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    expansion: QueryExpansion = Field(..., description="Query expansion suggestions")
    temporal_info: TemporalInfo = Field(..., description="Temporal analysis")
    modality_preference: ModalityPreference = Field(..., description="Modality preferences")
    
    # Query characteristics
    query_complexity: str = Field(..., description="simple, medium, complex")
    ambiguity_level: str = Field(..., description="low, medium, high")
    ambiguity_issues: List[str] = Field(default_factory=list, description="Identified ambiguities")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions to resolve ambiguity")
    
    # Search guidance
    search_type: str = Field(..., description="Recommended search type: keyword, semantic, hybrid, multimodal")
    search_filters: Dict[str, Any] = Field(default_factory=dict, description="Suggested search filters")
    
    # Processing info
    processing_time: float = Field(..., description="Understanding processing time")
    confidence_score: float = Field(..., description="Overall understanding confidence")
    status: str = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class QueryUnderstandingAgent(ConversationalAgent):
    """
    Agent chuyên hiểu và phân tích user queries
    Xử lý intent, entities, expansion, ambiguity resolution
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="QueryUnderstanding",
            model_type="gpt-4o",  # Cần model mạnh cho NLU
            **kwargs
        )
        
        self.set_instructions([
            "You are a query understanding and natural language processing specialist.",
            "Your role is to comprehensively analyze user queries for video search:",
            "- Classify user intent and sub-intent accurately",
            "- Extract named entities with proper typing",
            "- Expand queries with synonyms and related terms",
            "- Identify temporal references and constraints",
            "- Detect ambiguities and suggest clarifications",
            "- Recommend optimal search strategies",
            "- Understand multi-modal preferences",
            "Always provide confidence scores and detailed analysis.",
            "Focus on helping users find exactly what they're looking for."
        ])
        
        self.agent.response_model = QueryUnderstanding
        
    def process(self, user_query: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> AgentResponse:
        """
        Analyze và understand user query
        
        Args:
            user_query: User's natural language query
            context: Optional conversation context
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Create understanding prompt
            prompt = self._create_understanding_prompt(user_query, context, **kwargs)
            
            # Run agent với structured output
            response = self.agent.run(prompt, user_id=kwargs.get('user_id'))
            
            # Validate response
            if not isinstance(response, QueryUnderstanding):
                result = self._parse_understanding_response(response.content, user_query)
            else:
                result = response
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="query_understanding",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "query_length": len(user_query),
                    "entities_found": len(result.entities),
                    "ambiguity_level": result.ambiguity_level,
                    "search_type": result.search_type,
                    "confidence": result.confidence_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query understanding failed: {str(e)}")
            
            return self._create_response(
                task_type="query_understanding",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_understanding_prompt(self, user_query: str, context: Optional[Dict], **kwargs) -> str:
        """Tạo prompt cho query understanding"""
        
        prompt = f"""
        Analyze and understand this user query comprehensively: "{user_query}"
        
        ANALYSIS TASKS:
        
        1. INTENT CLASSIFICATION:
           - Identify main intent: search, explain, compare, list, navigate, summarize, etc.
           - Determine sub-intent for more specific categorization
           - Assess confidence level for intent classification
           - Provide clear description of what user wants to accomplish
        
        2. ENTITY EXTRACTION:
           - Extract all named entities: persons, objects, locations, actions, times
           - Provide normalized forms and known aliases
           - Specify entity types and positions in query
           - Include confidence scores for each entity
        
        3. QUERY EXPANSION:
           - Generate expanded query variations
           - Suggest relevant synonyms and related terms
           - Recommend search strategies for better recall
           - Consider different ways users might express the same intent
        
        4. TEMPORAL ANALYSIS:
           - Identify temporal references (times, dates, durations)
           - Parse time constraints and relative temporal expressions
           - Determine if query has temporal dependencies
        
        5. MODALITY PREFERENCES:
           - Detect preferred search modalities (video, audio, text, image)
           - Assign weights based on query context
           - Identify explicit modality requests
        
        6. AMBIGUITY DETECTION:
           - Identify potential ambiguities in the query
           - Generate clarification questions if needed
           - Assess overall query complexity
           - Suggest resolution strategies
        
        7. SEARCH STRATEGY:
           - Recommend optimal search type (keyword, semantic, hybrid, multimodal)
           - Suggest search filters and constraints
           - Provide search optimization guidance
        """
        
        # Add context if available
        if context:
            prompt += f"""
        
        CONVERSATION CONTEXT:
        - Previous queries: {context.get('previous_queries', [])}
        - User preferences: {context.get('user_preferences', {})}
        - Session history: {context.get('session_info', {})}
        - Current conversation topic: {context.get('current_topic', 'unknown')}
        
        Consider this context when analyzing the current query.
        """
        
        prompt += """
        
        EXAMPLES OF GOOD ANALYSIS:
        
        Query: "Show me cooking videos with pasta from last month"
        - Intent: search (specific_search)
        - Entities: [cooking(action), pasta(object), last month(time)]
        - Temporal: relative time constraint
        - Search type: hybrid (keyword + temporal filter)
        
        Query: "What's happening in this scene?"
        - Intent: explain (scene_description)
        - Entities: [scene(object)]
        - Ambiguity: high (no specific scene referenced)
        - Clarification needed: which scene/video?
        
        OUTPUT REQUIREMENTS:
        - Return complete QueryUnderstanding object
        - Include all analysis components with confidence scores
        - Provide actionable search recommendations
        - Identify ambiguities and suggest clarifications
        - Optimize for search effectiveness
        """
        
        return prompt
    
    def _parse_understanding_response(self, response_content: str, user_query: str) -> QueryUnderstanding:
        """Fallback parsing nếu structured output fails"""
        return QueryUnderstanding(
            query_id=f"query_{int(time.time())}",
            original_query=user_query,
            processed_query=user_query.lower().strip(),
            intent=Intent(
                intent_type="search",
                sub_intent="general_search",
                confidence=0.5,
                description="General search intent"
            ),
            entities=[],
            expansion=QueryExpansion(
                original_query=user_query,
                expanded_queries=[user_query],
                synonyms=[],
                related_terms=[],
                search_strategies=["keyword_search"]
            ),
            temporal_info=TemporalInfo(),
            modality_preference=ModalityPreference(
                preferred_modalities=["video"],
                modality_weights={"video": 1.0}
            ),
            query_complexity="medium",
            ambiguity_level="medium",
            ambiguity_issues=[],
            clarification_questions=[],
            search_type="keyword",
            search_filters={},
            processing_time=0,
            confidence_score=0.5,
            status="parsed_fallback"
        )
    
    def understand_batch_queries(self, queries: List[str], **kwargs) -> List[AgentResponse]:
        """Analyze multiple queries in batch"""
        results = []
        
        for query in queries:
            result = self.process(query, **kwargs)
            results.append(result)
            
            # Optional delay để tránh rate limiting
            if kwargs.get('delay_between_queries', 0) > 0:
                time.sleep(kwargs['delay_between_queries'])
        
        return results
    
    def clarify_ambiguous_query(self, understanding_result: Dict[str, Any], **kwargs) -> AgentResponse:
        """Generate clarification questions cho ambiguous queries"""
        ambiguity_level = understanding_result.get("ambiguity_level", "low")
        ambiguity_issues = understanding_result.get("ambiguity_issues", [])
        
        if ambiguity_level == "low":
            return self._create_response(
                task_type="clarification",
                status="success",
                result={"clarification_needed": False, "message": "Query is clear"},
                execution_time=0
            )
        
        prompt = f"""
        Generate helpful clarification questions for this ambiguous query understanding:
        
        Original Query: {understanding_result.get('original_query', '')}
        Ambiguity Level: {ambiguity_level}
        Ambiguity Issues: {ambiguity_issues}
        
        Create 2-3 specific questions that would help resolve the ambiguity.
        Make questions natural and helpful for the user.
        """
        
        response = self.run_with_timing(prompt, **kwargs)
        
        return self._create_response(
            task_type="clarification",
            status="success",
            result={
                "clarification_needed": True,
                "questions": response.result.get("content", "").split("\n"),
                "ambiguity_level": ambiguity_level
            },
            execution_time=response.execution_time
        )
    
    def expand_query_for_search(self, user_query: str, expansion_type: str = "comprehensive", **kwargs) -> AgentResponse:
        """Generate query expansions cho better search recall"""
        
        expansion_prompts = {
            "synonyms": "Generate synonym variations of the query terms",
            "related": "Generate related concepts and terms",
            "comprehensive": "Generate comprehensive query expansions including synonyms, related terms, and alternative phrasings",
            "multimodal": "Generate expansions optimized for multi-modal search"
        }
        
        prompt = f"""
        {expansion_prompts.get(expansion_type, expansion_prompts['comprehensive'])}
        
        Original Query: "{user_query}"
        
        Generate 5-10 query variations that would help find relevant content.
        Consider different ways users might express the same search intent.
        Include both specific and general variations.
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def detect_conversation_context(self, current_query: str, conversation_history: List[Dict], **kwargs) -> AgentResponse:
        """Detect conversation context và dependencies"""
        
        prompt = f"""
        Analyze conversation context for this query: "{current_query}"
        
        Conversation History:
        """
        
        for i, turn in enumerate(conversation_history[-5:]):  # Last 5 turns
            prompt += f"\nTurn {i+1}:\n"
            prompt += f"  User: {turn.get('user_message', '')}\n"
            prompt += f"  Assistant: {turn.get('assistant_message', '')[:100]}...\n"
        
        prompt += """
        
        Analyze:
        1. Does current query reference previous conversation?
        2. What context from history is relevant?
        3. Are there unresolved references (pronouns, "that video", etc.)?
        4. What's the evolving conversation topic?
        5. Should query be reformulated with context?
        
        Provide context-aware query understanding.
        """
        
        return self.run_with_timing(prompt, **kwargs)
    
    def get_intent_types(self) -> Dict[str, Any]:
        """Lấy danh sách supported intent types"""
        return {
            "search_intents": {
                "specific_search": "Search for specific content",
                "general_browse": "Browse content categories",
                "filtered_search": "Search with specific filters",
                "similar_search": "Find similar content"
            },
            "explanation_intents": {
                "scene_description": "Describe what's happening",
                "object_explanation": "Explain objects in video",
                "action_analysis": "Analyze actions/activities",
                "context_explanation": "Provide context/background"
            },
            "comparison_intents": {
                "compare_videos": "Compare multiple videos",
                "compare_scenes": "Compare scenes within video",
                "compare_objects": "Compare objects/people"
            },
            "navigation_intents": {
                "jump_to_time": "Navigate to specific time",
                "find_segment": "Find specific segment",
                "browse_sections": "Browse video sections"
            },
            "analysis_intents": {
                "summarize": "Summarize content",
                "analyze_trends": "Analyze patterns/trends",
                "extract_insights": "Extract key insights"
            }
        }
    
    def benchmark_understanding(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark query understanding performance"""
        results = []
        
        for query in test_queries:
            result = self.process(query)
            results.append({
                "query": query,
                "status": result.status,
                "processing_time": result.execution_time,
                "confidence": result.result.get("confidence_score", 0),
                "entities_found": len(result.result.get("entities", [])),
                "ambiguity_level": result.result.get("ambiguity_level", "unknown")
            })
        
        return {
            "total_queries": len(test_queries),
            "successful": len([r for r in results if r["status"] == "success"]),
            "avg_processing_time": sum(r["processing_time"] for r in results) / len(results),
            "avg_confidence": sum(r["confidence"] for r in results) / len(results),
            "avg_entities_per_query": sum(r["entities_found"] for r in results) / len(results),
            "ambiguity_distribution": {
                "low": len([r for r in results if r["ambiguity_level"] == "low"]),
                "medium": len([r for r in results if r["ambiguity_level"] == "medium"]),
                "high": len([r for r in results if r["ambiguity_level"] == "high"])
            },
            "details": results
        }