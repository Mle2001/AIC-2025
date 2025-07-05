"""
Video Retrieval Agent - Multi-modal search và ranking
Vector Search, Semantic Matching, Relevance Ranking, Result Fusion
Phần của Phase 2: Competition - Real-time video retrieval
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import time

from ..base_agent import ConversationalAgent, AgentResponse

# Structured output models
class SearchResult(BaseModel):
    """Individual search result"""
    result_id: str = Field(..., description="Unique result ID")
    video_id: str = Field(..., description="Source video ID")
    content_type: str = Field(..., description="Content type: video, scene, segment, frame")
    
    # Content info
    title: str = Field(..., description="Result title/description")
    description: str = Field(..., description="Detailed description")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail image URL")
    
    # Temporal info
    start_time: float = Field(..., description="Start timestamp in video")
    end_time: float = Field(..., description="End timestamp in video")
    duration: float = Field(..., description="Segment duration")
    
    # Relevance scores
    relevance_score: float = Field(..., description="Overall relevance score (0-1)")
    text_score: float = Field(default=0.0, description="Text matching score")
    visual_score: float = Field(default=0.0, description="Visual matching score")
    audio_score: float = Field(default=0.0, description="Audio matching score")
    semantic_score: float = Field(default=0.0, description="Semantic similarity score")
    
    # Matching info
    matched_terms: List[str] = Field(default_factory=list, description="Matched search terms")
    matched_entities: List[str] = Field(default_factory=list, description="Matched entities")
    matching_modalities: List[str] = Field(default_factory=list, description="Modalities that matched")
    
    # Content features
    extracted_text: str = Field(default="", description="Extracted text content")
    detected_objects: List[str] = Field(default_factory=list, description="Detected visual objects")
    audio_transcript: str = Field(default="", description="Audio transcript")
    
    # Metadata
    confidence: float = Field(..., description="Result confidence")
    source_indexes: List[str] = Field(default_factory=list, description="Source index IDs")

class SearchStrategy(BaseModel):
    """Search strategy configuration"""
    strategy_type: str = Field(..., description="Strategy type: keyword, semantic, hybrid, multimodal")
    
    # Search parameters
    use_text_search: bool = Field(default=True, description="Enable text search")
    use_visual_search: bool = Field(default=True, description="Enable visual search")
    use_audio_search: bool = Field(default=True, description="Enable audio search")
    use_semantic_search: bool = Field(default=True, description="Enable semantic search")
    
    # Weights
    text_weight: float = Field(default=0.3, description="Text search weight")
    visual_weight: float = Field(default=0.3, description="Visual search weight")
    audio_weight: float = Field(default=0.2, description="Audio search weight")
    semantic_weight: float = Field(default=0.2, description="Semantic search weight")
    
    # Filters
    time_range: Optional[Tuple[float, float]] = Field(None, description="Time range filter")
    min_duration: Optional[float] = Field(None, description="Minimum duration filter")
    max_duration: Optional[float] = Field(None, description="Maximum duration filter")
    content_types: List[str] = Field(default_factory=list, description="Content type filters")
    
    # Performance
    max_results: int = Field(default=20, description="Maximum results to return")
    min_score_threshold: float = Field(default=0.1, description="Minimum relevance score")

class RetrievalResult(BaseModel):
    """Comprehensive retrieval result"""
    query_id: str = Field(..., description="Query ID")
    original_query: str = Field(..., description="Original search query")
    processed_query: str = Field(..., description="Processed search query")
    
    # Search results
    results: List[SearchResult] = Field(default_factory=list, description="Ranked search results")
    total_found: int = Field(..., description="Total results found")
    returned_count: int = Field(..., description="Number of results returned")
    
    # Search strategy used
    strategy: SearchStrategy = Field(..., description="Search strategy used")
    
    # Search performance
    search_time: float = Field(..., description="Search execution time")
    ranking_time: float = Field(..., description="Ranking execution time")
    total_time: float = Field(..., description="Total retrieval time")
    
    # Search quality metrics
    avg_relevance_score: float = Field(..., description="Average relevance score")
    score_distribution: Dict[str, int] = Field(default_factory=dict, description="Score distribution")
    
    # Retrieval info
    indexes_searched: List[str] = Field(default_factory=list, description="Indexes searched")
    search_method: str = Field(..., description="Search method used")
    status: str = Field(..., description="Retrieval status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class VideoRetrievalAgent(ConversationalAgent):
    """
    Agent chuyên multi-modal video retrieval
    Thực hiện vector search, semantic matching, relevance ranking
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="VideoRetrieval",
            model_type="gpt-4o",
            **kwargs
        )
        
        self.set_instructions([
            "You are a multi-modal video retrieval and ranking specialist.",
            "Your role is to find the most relevant video content for user queries:",
            "- Perform multi-modal search across text, visual, and audio indexes",
            "- Combine keyword and semantic search for best results",
            "- Rank results by relevance using multiple scoring factors",
            "- Fuse results from different modalities intelligently",
            "- Apply temporal and content filters as needed",
            "- Optimize search strategies based on query characteristics",
            "Focus on high precision and recall for user satisfaction.",
            "Always provide detailed relevance explanations."
        ])
        
        self.agent.response_model = RetrievalResult
        
    def process(self, 
                query_understanding: Dict[str, Any],
                search_strategy: Optional[SearchStrategy] = None,
                **kwargs) -> AgentResponse:
        """
        Retrieve relevant video content based on understood query
        
        Args:
            query_understanding: Query understanding từ QueryUnderstandingAgent
            search_strategy: Optional search strategy override
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Extract query info
            original_query = query_understanding.get("original_query", "")
            processed_query = query_understanding.get("processed_query", original_query)
            
            # Determine search strategy
            if search_strategy is None:
                search_strategy = self._determine_search_strategy(query_understanding)
            
            # Create retrieval prompt
            prompt = self._create_retrieval_prompt(query_understanding, search_strategy)
            
            # Run agent với structured output
            response = self.agent.run(prompt, user_id=kwargs.get('user_id'))
            
            # Validate response
            if not isinstance(response, RetrievalResult):
                result = self._parse_retrieval_response(response.content, query_understanding, search_strategy)
            else:
                result = response
            
            # Post-process results để improve ranking
            result = self._rerank_results(result, query_understanding)
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="video_retrieval",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "query": original_query,
                    "results_found": result.total_found,
                    "results_returned": result.returned_count,
                    "search_strategy": search_strategy.strategy_type,
                    "avg_relevance": result.avg_relevance_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Video retrieval failed: {str(e)}")
            
            return self._create_response(
                task_type="video_retrieval",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _determine_search_strategy(self, query_understanding: Dict[str, Any]) -> SearchStrategy:
        """Determine optimal search strategy based on query characteristics"""
        
        # Extract query characteristics
        intent_type = query_understanding.get("intent", {}).get("intent_type", "search")
        query_complexity = query_understanding.get("query_complexity", "medium")
        entities = query_understanding.get("entities", [])
        temporal_info = query_understanding.get("temporal_info", {})
        modality_preference = query_understanding.get("modality_preference", {})
        
        # Base strategy
        strategy = SearchStrategy(strategy_type="hybrid")
        
        # Adjust based on intent
        if intent_type == "search":
            if query_complexity == "simple":
                strategy.strategy_type = "keyword"
                strategy.text_weight = 0.7
                strategy.semantic_weight = 0.3
            else:
                strategy.strategy_type = "hybrid"
                strategy.text_weight = 0.3
                strategy.semantic_weight = 0.4
                strategy.visual_weight = 0.2
                strategy.audio_weight = 0.1
        
        elif intent_type == "explain":
            strategy.strategy_type = "semantic"
            strategy.semantic_weight = 0.5
            strategy.visual_weight = 0.3
            strategy.text_weight = 0.2
        
        # Adjust based on modality preferences
        preferred_modalities = modality_preference.get("preferred_modalities", [])
        if "video" in preferred_modalities:
            strategy.visual_weight = min(strategy.visual_weight * 1.5, 0.5)
        if "audio" in preferred_modalities:
            strategy.audio_weight = min(strategy.audio_weight * 1.5, 0.4)
        
        # Apply temporal filters
        if temporal_info.get("has_temporal", False):
            time_constraints = temporal_info.get("time_constraints", {})
            if "start_time" in time_constraints and "end_time" in time_constraints:
                strategy.time_range = (
                    time_constraints["start_time"],
                    time_constraints["end_time"]
                )
        
        # Set entity-based filters
        entity_types = [e.get("entity_type") for e in entities]
        if "person" in entity_types:
            strategy.visual_weight = min(strategy.visual_weight * 1.2, 0.4)
        if "object" in entity_types:
            strategy.visual_weight = min(strategy.visual_weight * 1.3, 0.5)
        
        return strategy
    
    def _create_retrieval_prompt(self, query_understanding: Dict, strategy: SearchStrategy) -> str:
        """Tạo prompt cho retrieval process"""
        
        # Extract query components
        original_query = query_understanding.get("original_query", "")
        entities = query_understanding.get("entities", [])
        expansion = query_understanding.get("expansion", {})
        intent = query_understanding.get("intent", {})
        
        prompt = f"""
        Perform multi-modal video retrieval for query: "{original_query}"
        
        QUERY UNDERSTANDING:
        - Intent: {intent.get('intent_type', 'unknown')} ({intent.get('description', '')})
        - Entities: {[e.get('entity_text', '') for e in entities]}
        - Expanded terms: {expansion.get('related_terms', [])}
        - Query complexity: {query_understanding.get('query_complexity', 'medium')}
        
        SEARCH STRATEGY:
        - Strategy Type: {strategy.strategy_type}
        - Text Search: {strategy.use_text_search} (weight: {strategy.text_weight})
        - Visual Search: {strategy.use_visual_search} (weight: {strategy.visual_weight})
        - Audio Search: {strategy.use_audio_search} (weight: {strategy.audio_weight})
        - Semantic Search: {strategy.use_semantic_search} (weight: {strategy.semantic_weight})
        - Max Results: {strategy.max_results}
        - Min Score Threshold: {strategy.min_score_threshold}
        
        RETRIEVAL TASKS:
        """
        
        if strategy.use_text_search:
            prompt += f"""
        1. TEXT SEARCH:
           - Search text indexes (transcripts, OCR, descriptions)
           - Use query terms and expanded variations
           - Apply BM25/TF-IDF ranking for keyword matching
           - Weight: {strategy.text_weight}
        """
        
        if strategy.use_visual_search:
            prompt += f"""
        2. VISUAL SEARCH:
           - Search visual indexes (keyframes, objects, scenes)
           - Match visual entities and concepts
           - Use CLIP embeddings for semantic visual matching
           - Weight: {strategy.visual_weight}
        """
        
        if strategy.use_audio_search:
            prompt += f"""
        3. AUDIO SEARCH:
           - Search audio indexes (speech, sounds, music)
           - Match audio content and characteristics
           - Consider speaker and emotion information
           - Weight: {strategy.audio_weight}
        """
        
        if strategy.use_semantic_search:
            prompt += f"""
        4. SEMANTIC SEARCH:
           - Search semantic embeddings across all modalities
           - Find conceptually similar content
           - Use dense vector similarity search
           - Weight: {strategy.semantic_weight}
        """
        
        prompt += f"""
        
        RESULT FUSION AND RANKING:
        - Combine results from all enabled search methods
        - Calculate weighted relevance scores
        - Remove duplicates and near-duplicates
        - Apply temporal and content filters
        - Rank by overall relevance score
        - Return top {strategy.max_results} results above {strategy.min_score_threshold} threshold
        
        SEARCH EXECUTION:
        - Search all relevant indexes simultaneously
        - Measure search and ranking performance
        - Provide detailed relevance explanations
        - Include matched terms and entities
        - Generate comprehensive result metadata
        
        OUTPUT REQUIREMENTS:
        - Return complete RetrievalResult object
        - Include detailed SearchResult objects for each result
        - Provide relevance scores broken down by modality
        - Include search performance metrics
        - Explain matching and ranking decisions
        """
        
        return prompt
    
    def _parse_retrieval_response(self, response_content: str, query_understanding: Dict, strategy: SearchStrategy) -> RetrievalResult:
        """Fallback parsing nếu structured output fails"""
        return RetrievalResult(
            query_id=f"retrieval_{int(time.time())}",
            original_query=query_understanding.get("original_query", ""),
            processed_query=query_understanding.get("processed_query", ""),
            results=[],
            total_found=0,
            returned_count=0,
            strategy=strategy,
            search_time=0,
            ranking_time=0,
            total_time=0,
            avg_relevance_score=0,
            score_distribution={},
            indexes_searched=["fallback"],
            search_method="fallback_parser",
            status="parsed_fallback"
        )
    
    def _rerank_results(self, result: RetrievalResult, query_understanding: Dict) -> RetrievalResult:
        """Re-rank results để improve relevance"""
        
        if not result.results:
            return result
        
        # Extract query entities cho entity-based boosting
        query_entities = [e.get("entity_text", "").lower() for e in query_understanding.get("entities", [])]
        
        # Re-calculate scores với additional factors
        for search_result in result.results:
            # Entity matching bonus
            entity_bonus = 0
            for entity in query_entities:
                if entity in search_result.title.lower() or entity in search_result.description.lower():
                    entity_bonus += 0.1
                if entity in search_result.matched_entities:
                    entity_bonus += 0.15
            
            # Duration-based adjustment
            duration_factor = 1.0
            if search_result.duration < 5:  # Very short clips might be less useful
                duration_factor = 0.9
            elif search_result.duration > 300:  # Very long segments might be less precise
                duration_factor = 0.95
            
            # Multi-modal matching bonus
            modality_bonus = len(search_result.matching_modalities) * 0.05
            
            # Update relevance score
            search_result.relevance_score = min(
                search_result.relevance_score + entity_bonus + modality_bonus,
                1.0
            ) * duration_factor
        
        # Re-sort by updated scores
        result.results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update statistics
        result.avg_relevance_score = sum(r.relevance_score for r in result.results) / len(result.results)
        
        return result
    
    def search_by_text_only(self, text_query: str, max_results: int = 10, **kwargs) -> AgentResponse:
        """Search chỉ bằng text content"""
        # Create minimal query understanding
        query_understanding = {
            "original_query": text_query,
            "processed_query": text_query.lower().strip(),
            "intent": {"intent_type": "search", "description": "Text search"},
            "entities": [],
            "expansion": {"related_terms": []},
            "query_complexity": "simple"
        }
        
        # Text-only strategy
        strategy = SearchStrategy(
            strategy_type="keyword",
            use_text_search=True,
            use_visual_search=False,
            use_audio_search=False,
            use_semantic_search=False,
            text_weight=1.0,
            max_results=max_results
        )
        
        return self.process(query_understanding, strategy, **kwargs)
    
    def search_by_visual_similarity(self, visual_description: str, max_results: int = 10, **kwargs) -> AgentResponse:
        """Search bằng visual similarity"""
        query_understanding = {
            "original_query": visual_description,
            "processed_query": visual_description,
            "intent": {"intent_type": "search", "description": "Visual similarity search"},
            "entities": [],
            "modality_preference": {"preferred_modalities": ["video"]}
        }
        
        strategy = SearchStrategy(
            strategy_type="semantic",
            use_text_search=False,
            use_visual_search=True,
            use_audio_search=False,
            use_semantic_search=True,
            visual_weight=0.7,
            semantic_weight=0.3,
            max_results=max_results
        )
        
        return self.process(query_understanding, strategy, **kwargs)
    
    def search_multimodal(self, query_understanding: Dict, **kwargs) -> AgentResponse:
        """Full multi-modal search với all modalities"""
        strategy = SearchStrategy(
            strategy_type="multimodal",
            use_text_search=True,
            use_visual_search=True,
            use_audio_search=True,
            use_semantic_search=True,
            text_weight=0.25,
            visual_weight=0.25,
            audio_weight=0.25,
            semantic_weight=0.25,
            max_results=20
        )
        
        return self.process(query_understanding, strategy, **kwargs)
    
    def get_search_statistics(self, retrieval_results: List[Dict]) -> Dict[str, Any]:
        """Analyze search performance statistics"""
        if not retrieval_results:
            return {"error": "No results to analyze"}
        
        total_queries = len(retrieval_results)
        successful_searches = len([r for r in retrieval_results if r.get("status") == "success"])
        
        # Performance metrics
        search_times = [r.get("search_time", 0) for r in retrieval_results if r.get("search_time")]
        avg_search_time = sum(search_times) / len(search_times) if search_times else 0
        
        # Relevance metrics
        avg_relevances = [r.get("avg_relevance_score", 0) for r in retrieval_results]
        overall_avg_relevance = sum(avg_relevances) / len(avg_relevances) if avg_relevances else 0
        
        # Result counts
        result_counts = [r.get("returned_count", 0) for r in retrieval_results]
        avg_results_per_query = sum(result_counts) / len(result_counts) if result_counts else 0
        
        return {
            "total_queries": total_queries,
            "successful_searches": successful_searches,
            "success_rate": successful_searches / total_queries,
            "avg_search_time": round(avg_search_time, 3),
            "avg_relevance_score": round(overall_avg_relevance, 3),
            "avg_results_per_query": round(avg_results_per_query, 1),
            "performance_distribution": {
                "fast_searches": len([t for t in search_times if t < 0.5]),
                "medium_searches": len([t for t in search_times if 0.5 <= t < 2.0]),
                "slow_searches": len([t for t in search_times if t >= 2.0])
            }
        }
    
    def explain_retrieval_decision(self, result: SearchResult, query: str) -> str:
        """Explain why a result was retrieved and ranked"""
        explanation = f"Result '{result.title}' was retrieved for query '{query}' because:\n"
        
        # Relevance score breakdown
        explanation += f"- Overall relevance score: {result.relevance_score:.3f}\n"
        if result.text_score > 0:
            explanation += f"- Text matching score: {result.text_score:.3f}\n"
        if result.visual_score > 0:
            explanation += f"- Visual matching score: {result.visual_score:.3f}\n"
        if result.audio_score > 0:
            explanation += f"- Audio matching score: {result.audio_score:.3f}\n"
        if result.semantic_score > 0:
            explanation += f"- Semantic similarity score: {result.semantic_score:.3f}\n"
        
        # Matching details
        if result.matched_terms:
            explanation += f"- Matched terms: {', '.join(result.matched_terms)}\n"
        if result.matched_entities:
            explanation += f"- Matched entities: {', '.join(result.matched_entities)}\n"
        if result.matching_modalities:
            explanation += f"- Matching modalities: {', '.join(result.matching_modalities)}\n"
        
        # Content details
        explanation += f"- Time range: {result.start_time:.1f}s - {result.end_time:.1f}s ({result.duration:.1f}s)\n"
        explanation += f"- Confidence: {result.confidence:.3f}\n"
        
        return explanation