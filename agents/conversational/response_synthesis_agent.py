"""
Response Synthesis Agent - Tổng hợp và tạo final response
Answer Synthesis, Media Assembly, Response Ranking, Quality Check
Phần cuối của Phase 2: Competition - Final response generation
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import time

from ..base_agent import ConversationalAgent, AgentResponse

# Structured output models
class MediaReference(BaseModel):
    """Reference tới media content"""
    media_type: str = Field(..., description="Media type: video, image, audio, text")
    media_id: str = Field(..., description="Media ID")
    source_url: Optional[str] = Field(None, description="Source URL")
    
    # Temporal info
    start_time: Optional[float] = Field(None, description="Start timestamp")
    end_time: Optional[float] = Field(None, description="End timestamp")
    duration: Optional[float] = Field(None, description="Duration")
    
    # Display info
    title: str = Field(..., description="Display title")
    description: str = Field(..., description="Media description")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    
    # Relevance
    relevance_score: float = Field(..., description="Relevance to query")
    confidence: float = Field(..., description="Confidence in relevance")

class ResponseComponent(BaseModel):
    """Individual component của response"""
    component_id: str = Field(..., description="Component ID")
    component_type: str = Field(..., description="Type: text, media, list, explanation, etc.")
    
    # Content
    content: str = Field(..., description="Text content")
    media_references: List[MediaReference] = Field(default_factory=list, description="Associated media")
    
    # Structure
    priority: int = Field(..., description="Display priority (1=highest)")
    section: str = Field(..., description="Response section: main, supporting, additional")
    
    # Quality
    confidence: float = Field(..., description="Component confidence")
    completeness: float = Field(..., description="Completeness score")

class ResponseMetadata(BaseModel):
    """Metadata about response generation"""
    response_id: str = Field(..., description="Response ID")
    query_id: str = Field(..., description="Original query ID")
    
    # Sources
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    agent_sources: List[str] = Field(default_factory=list, description="Contributing agents")
    
    # Quality metrics
    overall_confidence: float = Field(..., description="Overall response confidence")
    completeness_score: float = Field(..., description="Response completeness")
    relevance_score: float = Field(..., description="Query relevance")
    
    # Processing info
    synthesis_time: float = Field(..., description="Synthesis processing time")
    component_count: int = Field(..., description="Number of components")
    media_count: int = Field(..., description="Number of media references")

class SynthesizedResponse(BaseModel):
    """Final synthesized response"""
    response_id: str = Field(..., description="Unique response ID")
    query: str = Field(..., description="Original user query")
    
    # Response content
    main_answer: str = Field(..., description="Primary response content")
    supporting_content: str = Field(default="", description="Supporting information")
    additional_info: str = Field(default="", description="Additional context")
    
    # Components
    components: List[ResponseComponent] = Field(default_factory=list, description="Response components")
    media_references: List[MediaReference] = Field(default_factory=list, description="Media content")
    
    # Response characteristics
    response_type: str = Field(..., description="Response type: answer, explanation, list, etc.")
    tone: str = Field(..., description="Response tone: formal, conversational, technical")
    detail_level: str = Field(..., description="Detail level: brief, medium, detailed")
    
    # User experience
    estimated_read_time: int = Field(..., description="Estimated reading time (seconds)")
    interactive_elements: List[str] = Field(default_factory=list, description="Interactive elements")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Follow-up question suggestions")
    
    # Quality and metadata
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    
    # Status
    status: str = Field(..., description="Synthesis status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class SynthesisRequest(BaseModel):
    """Request cho response synthesis"""
    query: str = Field(..., description="Original user query")
    query_understanding: Dict[str, Any] = Field(..., description="Query understanding results")
    
    # Input data
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    explanations: List[Dict[str, Any]] = Field(default_factory=list, description="Content explanations")
    context_info: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    
    # Synthesis preferences
    response_type: str = Field(default="comprehensive", description="Response type preference")
    detail_level: str = Field(default="medium", description="Desired detail level")
    include_media: bool = Field(default=True, description="Include media references")
    max_length: Optional[int] = Field(None, description="Maximum response length")
    
    # User preferences
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    communication_style: str = Field(default="conversational", description="Communication style")

class ResponseSynthesisAgent(ConversationalAgent):
    """
    Agent chuyên tổng hợp final response từ all sources
    Combines retrieval results, explanations, context để tạo coherent response
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ResponseSynthesis",
            model_type="gpt-4o",  # Cần model mạnh cho synthesis
            **kwargs
        )
        
        self.set_instructions([
            "You are a response synthesis and content assembly specialist.",
            "Your role is to create comprehensive, coherent responses by combining:",
            "- Search results from video retrieval",
            "- Content explanations and analysis",
            "- Conversation context and user preferences",
            "- Media references and supporting content",
            "Create responses that are:",
            "- Accurate and directly address the user's query",
            "- Well-structured and easy to understand",
            "- Appropriately detailed for the user's needs",
            "- Enhanced with relevant media and examples",
            "- Consistent with conversation context",
            "Always prioritize user satisfaction and clarity."
        ])
        
        self.agent.response_model = SynthesizedResponse
        
    def process(self, synthesis_request: SynthesisRequest, **kwargs) -> AgentResponse:
        """
        Synthesize comprehensive response từ all available data
        
        Args:
            synthesis_request: Request containing all input data
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(synthesis_request)
            
            # Run agent với structured output
            response = self.agent.run(prompt, user_id=kwargs.get('user_id'))
            
            # Validate response
            if not isinstance(response, SynthesizedResponse):
                result = self._parse_synthesis_response(response.content, synthesis_request)
            else:
                result = response
            
            # Post-process để enhance quality
            result = self._enhance_response_quality(result, synthesis_request)
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="response_synthesis",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "query": synthesis_request.query,
                    "response_length": len(result.main_answer),
                    "media_count": len(result.media_references),
                    "component_count": len(result.components),
                    "overall_confidence": result.metadata.overall_confidence
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Response synthesis failed: {str(e)}")
            
            return self._create_response(
                task_type="response_synthesis",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_synthesis_prompt(self, request: SynthesisRequest) -> str:
        """Tạo prompt cho response synthesis"""
        
        # Extract input data summaries
        retrieval_count = len(request.retrieval_results)
        explanation_count = len(request.explanations)
        user_intent = request.query_understanding.get("intent", {}).get("intent_type", "unknown")
        
        prompt = f"""
        Synthesize comprehensive response for query: "{request.query}"
        
        QUERY ANALYSIS:
        - Intent: {user_intent}
        - Complexity: {request.query_understanding.get('query_complexity', 'medium')}
        - Entities: {[e.get('entity_text') for e in request.query_understanding.get('entities', [])]}
        
        AVAILABLE DATA:
        - Retrieval Results: {retrieval_count} video search results
        - Content Explanations: {explanation_count} detailed explanations
        - Context Information: {len(request.context_info)} context elements
        
        SYNTHESIS REQUIREMENTS:
        - Response Type: {request.response_type}
        - Detail Level: {request.detail_level}
        - Communication Style: {request.communication_style}
        - Include Media: {request.include_media}
        - Max Length: {request.max_length or 'No limit'}
        """
        
        # Add retrieval results summary
        if request.retrieval_results:
            prompt += f"\nRETRIEVAL RESULTS:\n"
            for i, result in enumerate(request.retrieval_results[:5]):  # Top 5 results
                prompt += f"  Result {i+1} (score: {result.get('relevance_score', 0):.2f}):\n"
                prompt += f"    - Title: {result.get('title', 'N/A')}\n"
                prompt += f"    - Description: {result.get('description', 'N/A')[:100]}...\n"
                prompt += f"    - Time: {result.get('start_time', 0):.1f}-{result.get('end_time', 0):.1f}s\n"
                prompt += f"    - Content: {result.get('extracted_text', 'N/A')[:100]}...\n"
        
        # Add explanations summary
        if request.explanations:
            prompt += f"\nCONTENT EXPLANATIONS:\n"
            for i, explanation in enumerate(request.explanations[:3]):  # Top 3 explanations
                prompt += f"  Explanation {i+1}:\n"
                prompt += f"    - Type: {explanation.get('explanation_type', 'N/A')}\n"
                prompt += f"    - Summary: {explanation.get('overall_summary', 'N/A')[:100]}...\n"
                prompt += f"    - Key Insights: {explanation.get('key_insights', [])}\n"
        
        # Add context information
        if request.context_info:
            prompt += f"\nCONVERSATION CONTEXT:\n"
            prompt += f"    - Current Topic: {request.context_info.get('current_topic', 'None')}\n"
            prompt += f"    - Current Video: {request.context_info.get('current_video', 'None')}\n"
            prompt += f"    - User Preferences: {request.user_preferences}\n"
        
        prompt += f"""
        
        SYNTHESIS TASKS:
        
        1. CONTENT INTEGRATION:
           - Combine retrieval results and explanations coherently
           - Prioritize most relevant and high-confidence information
           - Resolve any conflicts or contradictions in sources
           - Ensure factual accuracy and consistency
        
        2. RESPONSE STRUCTURE:
           - Create clear main answer that directly addresses query
           - Organize supporting content logically
           - Include relevant examples and evidence
           - Structure for optimal readability
        
        3. MEDIA ASSEMBLY:
           - Select most relevant video segments and media
           - Provide proper timestamps and descriptions
           - Ensure media enhances rather than distracts
           - Include appropriate thumbnails and previews
        
        4. PERSONALIZATION:
           - Adapt tone and style to user preferences
           - Adjust detail level appropriately
           - Consider conversation context and history
           - Include relevant follow-up suggestions
        
        5. QUALITY ASSURANCE:
           - Ensure response directly answers the query
           - Verify all claims are supported by evidence
           - Check for completeness and clarity
           - Assess overall user satisfaction potential
        
        RESPONSE FORMATTING:
        - Main Answer: Direct, comprehensive response to query
        - Supporting Content: Additional context and details
        - Media References: Relevant video segments with timestamps
        - Follow-up Suggestions: Natural next questions user might ask
        
        OUTPUT REQUIREMENTS:
        - Return complete SynthesizedResponse object
        - Include all components with proper prioritization
        - Provide comprehensive metadata and quality metrics
        - Ensure response is ready for immediate delivery
        """
        
        return prompt
    
    def _parse_synthesis_response(self, response_content: str, request: SynthesisRequest) -> SynthesizedResponse:
        """Fallback parsing nếu structured output fails"""
        return SynthesizedResponse(
            response_id=f"response_{int(time.time())}",
            query=request.query,
            main_answer="I found some relevant information for your query, but encountered an issue processing the complete response.",
            supporting_content="Please try rephrasing your question or asking for more specific information.",
            additional_info="",
            components=[],
            media_references=[],
            response_type="partial",
            tone="conversational",
            detail_level="brief",
            estimated_read_time=10,
            interactive_elements=[],
            follow_up_suggestions=["Could you be more specific?", "What aspect interests you most?"],
            metadata=ResponseMetadata(
                response_id=f"response_{int(time.time())}",
                query_id=request.query_understanding.get("query_id", "unknown"),
                data_sources=["fallback_parser"],
                agent_sources=["response_synthesis"],
                overall_confidence=0.5,
                completeness_score=0.3,
                relevance_score=0.4,
                synthesis_time=0,
                component_count=0,
                media_count=0
            ),
            status="parsed_fallback"
        )
    
    def _enhance_response_quality(self, response: SynthesizedResponse, request: SynthesisRequest) -> SynthesizedResponse:
        """Enhance response quality và completeness"""
        
        # Calculate estimated read time based on content length
        total_text_length = len(response.main_answer) + len(response.supporting_content) + len(response.additional_info)
        words_per_minute = 200
        response.estimated_read_time = max(int(total_text_length / (words_per_minute * 5)), 5)  # 5 chars per word average
        
        # Update metadata
        response.metadata.component_count = len(response.components)
        response.metadata.media_count = len(response.media_references)
        
        # Calculate quality scores
        content_quality = self._assess_content_quality(response, request)
        response.metadata.overall_confidence = content_quality["confidence"]
        response.metadata.completeness_score = content_quality["completeness"]
        response.metadata.relevance_score = content_quality["relevance"]
        
        # Add interactive elements based on content
        if response.media_references:
            response.interactive_elements.extend(["video_player", "timestamp_navigation"])
        
        if len(response.components) > 1:
            response.interactive_elements.append("section_navigation")
        
        # Generate follow-up suggestions if none exist
        if not response.follow_up_suggestions:
            response.follow_up_suggestions = self._generate_followup_suggestions(response, request)
        
        return response
    
    def _assess_content_quality(self, response: SynthesizedResponse, request: SynthesisRequest) -> Dict[str, float]:
        """Assess response content quality"""
        
        # Confidence based on content length and structure
        content_length = len(response.main_answer)
        confidence = min(content_length / 500, 1.0)  # Normalize to 500 chars for full confidence
        
        # Completeness based on addressing query components
        query_entities = request.query_understanding.get("entities", [])
        entity_coverage = 0
        if query_entities:
            covered_entities = sum(1 for entity in query_entities 
                                 if entity.get("entity_text", "").lower() in response.main_answer.lower())
            entity_coverage = covered_entities / len(query_entities)
        
        completeness = (0.5 + entity_coverage * 0.5)  # Base 0.5 + entity coverage
        
        # Relevance based on query intent matching
        intent_type = request.query_understanding.get("intent", {}).get("intent_type", "search")
        intent_keywords = {
            "search": ["found", "results", "videos", "content"],
            "explain": ["explanation", "describes", "shows", "analysis"],
            "compare": ["comparison", "differences", "similarities", "versus"],
            "summarize": ["summary", "overview", "key points", "main"]
        }
        
        relevance = 0.7  # Base relevance
        if intent_type in intent_keywords:
            matching_keywords = sum(1 for keyword in intent_keywords[intent_type]
                                  if keyword in response.main_answer.lower())
            relevance += (matching_keywords / len(intent_keywords[intent_type])) * 0.3
        
        return {
            "confidence": min(confidence, 1.0),
            "completeness": min(completeness, 1.0),
            "relevance": min(relevance, 1.0)
        }
    
    def _generate_followup_suggestions(self, response: SynthesizedResponse, request: SynthesisRequest) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions = []
        
        intent_type = request.query_understanding.get("intent", {}).get("intent_type", "search")
        
        if intent_type == "search":
            suggestions.extend([
                "Can you show me more details about any of these results?",
                "Are there similar videos on this topic?",
                "What's the most relevant part of these videos?"
            ])
        elif intent_type == "explain":
            suggestions.extend([
                "Can you explain this in more detail?",
                "What's the context behind this?",
                "Are there related concepts I should know about?"
            ])
        elif intent_type == "compare":
            suggestions.extend([
                "What are the key differences?",
                "Which option would you recommend?",
                "Can you compare other aspects?"
            ])
        
        # Add media-specific suggestions if media is present
        if response.media_references:
            suggestions.append("Can you play a specific segment?")
            suggestions.append("Show me the most relevant timestamp")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def synthesize_simple_answer(self, query: str, top_result: Dict[str, Any], **kwargs) -> AgentResponse:
        """Synthesize simple answer từ single top result"""
        
        simple_request = SynthesisRequest(
            query=query,
            query_understanding={"intent": {"intent_type": "search"}, "entities": []},
            retrieval_results=[top_result],
            explanations=[],
            context_info={},
            response_type="brief",
            detail_level="brief"
        )
        
        return self.process(simple_request, **kwargs)
    
    def synthesize_explanation_response(self, query: str, explanation_data: Dict[str, Any], **kwargs) -> AgentResponse:
        """Synthesize response focused on explanation"""
        
        explanation_request = SynthesisRequest(
            query=query,
            query_understanding={"intent": {"intent_type": "explain"}, "entities": []},
            retrieval_results=[],
            explanations=[explanation_data],
            context_info={},
            response_type="explanation",
            detail_level="detailed"
        )
        
        return self.process(explanation_request, **kwargs)
    
    def synthesize_comparison_response(self, query: str, results: List[Dict[str, Any]], **kwargs) -> AgentResponse:
        """Synthesize response for comparison queries"""
        
        comparison_request = SynthesisRequest(
            query=query,
            query_understanding={"intent": {"intent_type": "compare"}, "entities": []},
            retrieval_results=results,
            explanations=[],
            context_info={},
            response_type="comparison",
            detail_level="medium"
        )
        
        return self.process(comparison_request, **kwargs)
    
    def rank_response_quality(self, responses: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """
        Rank multiple response options by quality
        
        Args:
            responses: List of synthesized responses
            
        Returns:
            List of (index, quality_score) tuples sorted by quality
        """
        scored_responses = []
        
        for i, response in enumerate(responses):
            metadata = response.get("metadata", {})
            
            # Calculate combined quality score
            confidence = metadata.get("overall_confidence", 0.5)
            completeness = metadata.get("completeness_score", 0.5)
            relevance = metadata.get("relevance_score", 0.5)
            
            # Weight factors
            quality_score = (
                confidence * 0.4 +
                completeness * 0.3 +
                relevance * 0.3
            )
            
            # Bonus for media richness
            media_count = metadata.get("media_count", 0)
            if media_count > 0:
                quality_score += min(media_count * 0.1, 0.2)
            
            # Penalty for errors
            if response.get("status") != "success":
                quality_score *= 0.5
            
            scored_responses.append((i, quality_score))
        
        # Sort by quality score descending
        return sorted(scored_responses, key=lambda x: x[1], reverse=True)
    
    def get_synthesis_capabilities(self) -> Dict[str, Any]:
        """Lấy thông tin synthesis capabilities"""
        return {
            "response_types": {
                "brief": "Concise answers with essential information",
                "comprehensive": "Detailed responses with full context",
                "explanation": "Focus on explaining concepts and content",
                "comparison": "Comparative analysis of multiple items",
                "list": "Structured lists and enumeration",
                "narrative": "Story-like explanations with flow"
            },
            "synthesis_sources": [
                "video_search_results",
                "content_explanations",
                "conversation_context",
                "user_preferences",
                "knowledge_graph_data",
                "temporal_information"
            ],
            "media_integration": {
                "video_clips": "Relevant video segments with timestamps",
                "keyframes": "Representative frame images",
                "audio_segments": "Speech and sound excerpts",
                "text_overlays": "OCR and caption content"
            },
            "personalization_factors": [
                "communication_style",
                "detail_preference",
                "content_interests",
                "technical_level",
                "conversation_history"
            ],
            "quality_metrics": {
                "confidence": "Reliability of information",
                "completeness": "Coverage of query aspects",
                "relevance": "Match to user intent",
                "clarity": "Understandability",
                "engagement": "User satisfaction potential"
            }
        }