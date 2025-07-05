"""
Content Explainer Agent - Giải thích và mô tả video content
Scene Description, Object Explanation, Action Analysis, Detail Enhancement
Phần của Phase 2: Competition - Real-time content explanation
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time

from ..base_agent import ConversationalAgent, AgentResponse

# Structured output models
class SceneDescription(BaseModel):
    """Mô tả chi tiết scene"""
    scene_id: str = Field(..., description="Scene ID")
    timestamp: float = Field(..., description="Timestamp in video")
    
    # Basic description
    title: str = Field(..., description="Scene title/summary")
    description: str = Field(..., description="Detailed scene description")
    setting: str = Field(..., description="Scene setting/location")
    mood: str = Field(..., description="Scene mood/atmosphere")
    
    # Visual elements
    objects: List[Dict[str, Any]] = Field(default_factory=list, description="Objects in scene")
    people: List[Dict[str, Any]] = Field(default_factory=list, description="People in scene")
    actions: List[str] = Field(default_factory=list, description="Actions happening")
    
    # Technical details
    lighting: str = Field(..., description="Lighting conditions")
    camera_angle: str = Field(..., description="Camera angle/perspective")
    composition: str = Field(..., description="Visual composition")
    colors: List[str] = Field(default_factory=list, description="Dominant colors")

class ObjectExplanation(BaseModel):
    """Giải thích về objects trong video"""
    object_id: str = Field(..., description="Object ID")
    object_name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Object category")
    
    # Description
    description: str = Field(..., description="Detailed object description")
    purpose: str = Field(..., description="Object purpose/function")
    context: str = Field(..., description="Object context in scene")
    
    # Properties
    size: str = Field(..., description="Relative size description")
    position: str = Field(..., description="Position in frame")
    condition: str = Field(..., description="Object condition/state")
    
    # Temporal info
    appears_at: float = Field(..., description="First appearance time")
    duration: float = Field(..., description="Duration in video")
    key_moments: List[float] = Field(default_factory=list, description="Key interaction moments")

class ActionAnalysis(BaseModel):
    """Phân tích actions và activities"""
    action_id: str = Field(..., description="Action ID")
    action_type: str = Field(..., description="Action type/category")
    action_name: str = Field(..., description="Action name")
    
    # Description
    description: str = Field(..., description="Detailed action description")
    purpose: str = Field(..., description="Action purpose/goal")
    context: str = Field(..., description="Action context")
    
    # Participants
    primary_actor: Optional[str] = Field(None, description="Main person doing action")
    secondary_actors: List[str] = Field(default_factory=list, description="Other participants")
    affected_objects: List[str] = Field(default_factory=list, description="Objects involved")
    
    # Temporal info
    start_time: float = Field(..., description="Action start time")
    end_time: float = Field(..., description="Action end time")
    duration: float = Field(..., description="Action duration")
    
    # Characteristics
    intensity: str = Field(..., description="Action intensity level")
    complexity: str = Field(..., description="Action complexity")
    success: str = Field(..., description="Action outcome/success")

class ContextualExplanation(BaseModel):
    """Contextual explanation và background info"""
    topic: str = Field(..., description="Explanation topic")
    category: str = Field(..., description="Category: cultural, technical, historical, etc.")
    
    # Content
    summary: str = Field(..., description="Brief summary")
    detailed_explanation: str = Field(..., description="Detailed explanation")
    background_info: str = Field(..., description="Background context")
    
    # Related info
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    similar_examples: List[str] = Field(default_factory=list, description="Similar examples")
    additional_resources: List[str] = Field(default_factory=list, description="Additional resources")
    
    # Relevance
    relevance_to_video: str = Field(..., description="How this relates to video content")
    importance_level: str = Field(..., description="Importance level: high, medium, low")

class ComprehensiveExplanation(BaseModel):
    """Tổng hợp explanation result"""
    explanation_id: str = Field(..., description="Unique explanation ID")
    video_id: str = Field(..., description="Source video ID")
    explanation_type: str = Field(..., description="Type: scene, object, action, context, full")
    
    # Target content
    target_timestamp: Optional[float] = Field(None, description="Target timestamp")
    target_segment: Optional[Tuple[float, float]] = Field(None, description="Target time segment")
    target_entities: List[str] = Field(default_factory=list, description="Target entities to explain")
    
    # Explanations
    scene_descriptions: List[SceneDescription] = Field(default_factory=list, description="Scene descriptions")
    object_explanations: List[ObjectExplanation] = Field(default_factory=list, description="Object explanations")
    action_analyses: List[ActionAnalysis] = Field(default_factory=list, description="Action analyses")
    contextual_explanations: List[ContextualExplanation] = Field(default_factory=list, description="Contextual explanations")
    
    # Summary
    overall_summary: str = Field(..., description="Overall explanation summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    interesting_details: List[str] = Field(default_factory=list, description="Interesting details")
    
    # Quality metrics
    explanation_quality: str = Field(..., description="Quality assessment: excellent, good, fair")
    completeness_score: float = Field(..., description="Completeness score (0-1)")
    clarity_score: float = Field(..., description="Clarity score (0-1)")
    
    # Processing info
    processing_time: float = Field(..., description="Explanation generation time")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    status: str = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class ExplanationRequest(BaseModel):
    """Request cho content explanation"""
    request_type: str = Field(..., description="Type: scene, object, action, context, full")
    
    # Target specification
    video_id: str = Field(..., description="Target video ID")
    timestamp: Optional[float] = Field(None, description="Specific timestamp")
    time_range: Optional[Tuple[float, float]] = Field(None, description="Time range")
    entity_names: List[str] = Field(default_factory=list, description="Specific entities to explain")
    
    # Explanation preferences
    detail_level: str = Field(default="medium", description="Detail level: brief, medium, detailed")
    include_background: bool = Field(default=True, description="Include background context")
    include_technical: bool = Field(default=False, description="Include technical details")
    
    # Audience
    target_audience: str = Field(default="general", description="Target audience: general, expert, child")
    language_style: str = Field(default="conversational", description="Language style: formal, conversational, technical")

class ContentExplainerAgent(ConversationalAgent):
    """
    Agent chuyên giải thích và mô tả video content
    Cung cấp detailed explanations về scenes, objects, actions
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ContentExplainer",
            model_type="gpt-4o",  # Cần model mạnh cho detailed explanations
            **kwargs
        )
        
        self.set_instructions([
            "You are a content explanation and analysis specialist.",
            "Your role is to provide comprehensive explanations of video content:",
            "- Describe scenes with rich visual and contextual details",
            "- Explain objects, their purpose, and significance",
            "- Analyze actions, activities, and their meanings",
            "- Provide cultural, technical, or historical context when relevant",
            "- Adapt explanations to user's knowledge level and interests",
            "- Use clear, engaging language that enhances understanding",
            "Always be accurate, informative, and helpful.",
            "Focus on making complex content accessible and interesting."
        ])
        
        self.agent.response_model = ComprehensiveExplanation
        
    def process(self, 
                explanation_request: ExplanationRequest,
                content_data: Dict[str, Any],
                **kwargs) -> AgentResponse:
        """
        Generate comprehensive explanation cho requested content
        
        Args:
            explanation_request: What user wants explained
            content_data: Video content data (features, scenes, etc.)
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Create explanation prompt
            prompt = self._create_explanation_prompt(explanation_request, content_data)
            
            # Run agent với structured output
            response = self.agent.run(prompt, user_id=kwargs.get('user_id'))
            
            # Validate response
            if not isinstance(response, ComprehensiveExplanation):
                result = self._parse_explanation_response(response.content, explanation_request)
            else:
                result = response
            
            # Enhance explanation quality
            result = self._enhance_explanation(result, explanation_request, content_data)
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="content_explanation",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "explanation_type": explanation_request.request_type,
                    "detail_level": explanation_request.detail_level,
                    "completeness_score": result.completeness_score,
                    "clarity_score": result.clarity_score,
                    "components_count": {
                        "scenes": len(result.scene_descriptions),
                        "objects": len(result.object_explanations),
                        "actions": len(result.action_analyses),
                        "contexts": len(result.contextual_explanations)
                    }
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Content explanation failed: {str(e)}")
            
            return self._create_response(
                task_type="content_explanation",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_explanation_prompt(self, request: ExplanationRequest, content_data: Dict) -> str:
        """Tạo prompt cho content explanation"""
        
        # Extract relevant content data
        visual_features = content_data.get("visual_features", [])
        audio_features = content_data.get("audio_features", [])
        scenes = content_data.get("scenes", [])
        entities = content_data.get("entities", [])
        
        prompt = f"""
        Generate comprehensive content explanation for video: {request.video_id}
        
        EXPLANATION REQUEST:
        - Type: {request.request_type}
        - Detail Level: {request.detail_level}
        - Target Audience: {request.target_audience}
        - Language Style: {request.language_style}
        - Include Background: {request.include_background}
        - Include Technical: {request.include_technical}
        """
        
        # Add target specification
        if request.timestamp:
            prompt += f"\n- Target Timestamp: {request.timestamp}s"
        if request.time_range:
            prompt += f"\n- Target Time Range: {request.time_range[0]}s - {request.time_range[1]}s"
        if request.entity_names:
            prompt += f"\n- Target Entities: {request.entity_names}"
        
        prompt += f"""
        
        AVAILABLE CONTENT DATA:
        - Visual Features: {len(visual_features)} keyframes with descriptions
        - Audio Features: {len(audio_features)} segments with transcripts
        - Scene Information: {len(scenes)} detected scenes
        - Extracted Entities: {len(entities)} entities
        """
        
        # Add relevant content details based on request
        if request.timestamp or request.time_range:
            # Filter content for specific time range
            relevant_visual = self._filter_by_time(visual_features, request.timestamp, request.time_range)
            relevant_audio = self._filter_by_time(audio_features, request.timestamp, request.time_range)
            
            prompt += f"\nRELEVANT CONTENT FOR TARGET TIME:\n"
            
            # Visual content
            if relevant_visual:
                prompt += "VISUAL CONTENT:\n"
                for i, vf in enumerate(relevant_visual[:3]):
                    prompt += f"  Frame {i+1} (t={vf.get('timestamp', 0)}s):\n"
                    prompt += f"    - Description: {vf.get('blip_caption', 'N/A')}\n"
                    prompt += f"    - Objects: {vf.get('blip_objects', [])}\n"
                    prompt += f"    - Scene Type: {vf.get('scene_type', 'N/A')}\n"
            
            # Audio content
            if relevant_audio:
                prompt += "\nAUDIO CONTENT:\n"
                for i, af in enumerate(relevant_audio[:3]):
                    prompt += f"  Segment {i+1} (t={af.get('start_time', 0)}-{af.get('end_time', 0)}s):\n"
                    prompt += f"    - Transcript: {af.get('transcript', 'N/A')}\n"
                    prompt += f"    - Audio Type: {af.get('audio_type', 'N/A')}\n"
                    prompt += f"    - Emotion: {af.get('emotion', 'N/A')}\n"
        else:
            # General content overview
            prompt += f"\nGENERAL CONTENT OVERVIEW:\n"
            if visual_features:
                prompt += f"- Visual: {len(visual_features)} keyframes covering various scenes\n"
            if audio_features:
                prompt += f"- Audio: {len(audio_features)} audio segments with speech/sounds\n"
        
        # Add explanation tasks based on request type
        prompt += f"\nEXPLANATION TASKS:\n"
        
        if request.request_type in ["scene", "full"]:
            prompt += """
        1. SCENE DESCRIPTION:
           - Provide rich visual descriptions of scenes
           - Describe setting, mood, and atmosphere
           - Identify key visual elements and composition
           - Explain lighting and camera perspective
           - Note dominant colors and visual style
        """
        
        if request.request_type in ["object", "full"]:
            prompt += """
        2. OBJECT EXPLANATION:
           - Identify and describe significant objects
           - Explain object purpose and significance
           - Describe object properties and condition
           - Analyze object placement and context
           - Note object interactions and relationships
        """
        
        if request.request_type in ["action", "full"]:
            prompt += """
        3. ACTION ANALYSIS:
           - Identify and analyze activities/actions
           - Describe action purpose and context
           - Identify participants and their roles
           - Analyze action complexity and success
           - Explain action significance or meaning
        """
        
        if request.include_background:
            prompt += """
        4. CONTEXTUAL EXPLANATION:
           - Provide relevant background information
           - Explain cultural, historical, or technical context
           - Connect content to broader knowledge
           - Identify related concepts and examples
           - Assess importance and relevance
        """
        
        # Adaptation guidelines
        prompt += f"""
        
        ADAPTATION GUIDELINES:
        - Target Audience: {request.target_audience}
          - General: Use accessible language, avoid jargon
          - Expert: Include technical details and nuanced analysis
          - Child: Use simple language and relatable examples
        
        - Detail Level: {request.detail_level}
          - Brief: Focus on key points and main insights
          - Medium: Provide balanced detail and context
          - Detailed: Include comprehensive analysis and background
        
        - Language Style: {request.language_style}
          - Conversational: Use natural, engaging tone
          - Formal: Use professional, structured language
          - Technical: Include precise terminology and specifications
        
        OUTPUT REQUIREMENTS:
        - Return complete ComprehensiveExplanation object
        - Include all relevant explanation components
        - Provide clear, engaging descriptions
        - Adapt to specified audience and style
        - Include quality assessment and insights
        """
        
        return prompt
    
    def _filter_by_time(self, features: List[Dict], timestamp: Optional[float], time_range: Optional[Tuple[float, float]]) -> List[Dict]:
        """Filter features by time constraints"""
        if not features:
            return []
        
        if timestamp is not None:
            # Find features near specific timestamp (±5 seconds)
            return [f for f in features 
                   if abs(f.get('timestamp', 0) - timestamp) <= 5]
        
        elif time_range is not None:
            start_time, end_time = time_range
            return [f for f in features 
                   if start_time <= f.get('timestamp', 0) <= end_time or
                      start_time <= f.get('start_time', 0) <= end_time]
        
        return features
    
    def _parse_explanation_response(self, response_content: str, request: ExplanationRequest) -> ComprehensiveExplanation:
        """Fallback parsing nếu structured output fails"""
        return ComprehensiveExplanation(
            explanation_id=f"explanation_{int(time.time())}",
            video_id=request.video_id,
            explanation_type=request.request_type,
            target_timestamp=request.timestamp,
            target_segment=request.time_range,
            target_entities=request.entity_names,
            scene_descriptions=[],
            object_explanations=[],
            action_analyses=[],
            contextual_explanations=[],
            overall_summary="Explanation processing completed with fallback parser.",
            key_insights=[],
            interesting_details=[],
            explanation_quality="fair",
            completeness_score=0.5,
            clarity_score=0.5,
            processing_time=0,
            data_sources=["fallback_parser"],
            status="parsed_fallback"
        )
    
    def _enhance_explanation(self, result: ComprehensiveExplanation, request: ExplanationRequest, content_data: Dict) -> ComprehensiveExplanation:
        """Enhance explanation quality"""
        
        # Calculate quality scores based on content richness
        component_count = (
            len(result.scene_descriptions) +
            len(result.object_explanations) +
            len(result.action_analyses) +
            len(result.contextual_explanations)
        )
        
        # Completeness score based on components and detail
        if request.request_type == "full":
            expected_components = 4
        else:
            expected_components = 1
        
        result.completeness_score = min(component_count / expected_components, 1.0)
        
        # Clarity score based on description lengths and structure
        total_description_length = len(result.overall_summary)
        for desc in result.scene_descriptions:
            total_description_length += len(desc.description)
        
        if total_description_length > 500:
            result.clarity_score = 0.9
        elif total_description_length > 200:
            result.clarity_score = 0.7
        else:
            result.clarity_score = 0.5
        
        # Overall quality assessment
        avg_score = (result.completeness_score + result.clarity_score) / 2
        if avg_score >= 0.8:
            result.explanation_quality = "excellent"
        elif avg_score >= 0.6:
            result.explanation_quality = "good"
        else:
            result.explanation_quality = "fair"
        
        return result
    
    def explain_scene_at_timestamp(self, video_id: str, timestamp: float, detail_level: str = "medium", **kwargs) -> AgentResponse:
        """Explain specific scene at timestamp"""
        request = ExplanationRequest(
            request_type="scene",
            video_id=video_id,
            timestamp=timestamp,
            detail_level=detail_level
        )
        
        # Mock content data - trong thực tế sẽ lấy từ database
        content_data = {
            "visual_features": [],
            "audio_features": [],
            "scenes": [],
            "entities": []
        }
        
        return self.process(request, content_data, **kwargs)
    
    def explain_objects_in_range(self, video_id: str, start_time: float, end_time: float, **kwargs) -> AgentResponse:
        """Explain objects trong time range"""
        request = ExplanationRequest(
            request_type="object",
            video_id=video_id,
            time_range=(start_time, end_time),
            detail_level="detailed"
        )
        
        content_data = {
            "visual_features": [],
            "audio_features": [],
            "scenes": [],
            "entities": []
        }
        
        return self.process(request, content_data, **kwargs)
    
    def explain_actions_for_entities(self, video_id: str, entity_names: List[str], **kwargs) -> AgentResponse:
        """Explain actions involving specific entities"""
        request = ExplanationRequest(
            request_type="action",
            video_id=video_id,
            entity_names=entity_names,
            detail_level="detailed"
        )
        
        content_data = {
            "visual_features": [],
            "audio_features": [],
            "scenes": [],
            "entities": []
        }
        
        return self.process(request, content_data, **kwargs)
    
    def provide_context_explanation(self, topic: str, video_id: str, **kwargs) -> AgentResponse:
        """Provide contextual explanation về specific topic"""
        request = ExplanationRequest(
            request_type="context",
            video_id=video_id,
            entity_names=[topic],
            include_background=True,
            include_technical=kwargs.get('include_technical', False)
        )
        
        content_data = {
            "visual_features": [],
            "audio_features": [],
            "scenes": [],
            "entities": []
        }
        
        return self.process(request, content_data, **kwargs)
    
    def get_explanation_capabilities(self) -> Dict[str, Any]:
        """Lấy thông tin explanation capabilities"""
        return {
            "explanation_types": {
                "scene": "Detailed scene descriptions with visual analysis",
                "object": "Object identification and explanation",
                "action": "Action and activity analysis",
                "context": "Background and contextual information",
                "full": "Comprehensive multi-aspect explanation"
            },
            "detail_levels": {
                "brief": "Key points and main insights only",
                "medium": "Balanced detail with context",
                "detailed": "Comprehensive analysis with background"
            },
            "target_audiences": {
                "general": "Accessible explanations for general users",
                "expert": "Technical details for domain experts",
                "child": "Simple language for young users"
            },
            "content_aspects": [
                "visual_elements",
                "audio_content",
                "temporal_events",
                "spatial_relationships",
                "cultural_context",
                "technical_details",
                "emotional_content",
                "narrative_structure"
            ],
            "quality_metrics": {
                "completeness": "Coverage of requested aspects",
                "clarity": "Language clarity and structure",
                "accuracy": "Factual correctness",
                "relevance": "Relevance to user request"
            }
        }