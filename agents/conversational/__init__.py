"""
Conversational Agents Package - Phase 2 of AI Challenge System

This subpackage contains all agents responsible for Phase 2 real-time conversation:
- Understanding what users are asking for in natural language
- Finding relevant video content through multi-modal search
- Explaining video content in clear, helpful ways  
- Managing conversation context and user memory
- Synthesizing comprehensive responses with media

Think of these agents as a team of specialists working together to have intelligent
conversations about video content. Each agent has a specific expertise, and they
collaborate to provide the best possible user experience.
"""

# Import all conversational agents and their key components
from .query_understanding_agent import (
    QueryUnderstandingAgent,
    QueryUnderstanding,
    Intent,
    Entity,
    QueryExpansion,
    TemporalInfo,
    ModalityPreference
)

from .retrieval_agent import (
    VideoRetrievalAgent,
    RetrievalResult,
    SearchResult,
    SearchStrategy
)

from .explanation_agent import (
    ContentExplainerAgent,
    ComprehensiveExplanation,
    ExplanationRequest,
    SceneDescription,
    ObjectExplanation,
    ActionAnalysis,
    ContextualExplanation
)

from .context_manager_agent import (
    ContextManagerAgent,
    SessionContext,
    ConversationTurn,
    UserPreferences,
    ContextUpdate,
    ContextAnalysis
)

from .response_synthesis_agent import (
    ResponseSynthesisAgent,
    SynthesizedResponse,
    SynthesisRequest,
    ResponseComponent,
    MediaReference,
    ResponseMetadata
)

# Export all the important classes for easy importing
__all__ = [
    # Main agent classes - these are the core conversation specialists
    "QueryUnderstandingAgent",    # Understands what users are asking
    "VideoRetrievalAgent",        # Finds relevant video content  
    "ContentExplainerAgent",      # Explains video content clearly
    "ContextManagerAgent",        # Remembers conversation history
    "ResponseSynthesisAgent",     # Puts together final responses
    
    # Request and configuration classes - these help configure agent behavior
    "ExplanationRequest",         # Configures what to explain
    "SynthesisRequest",          # Configures response generation
    "SearchStrategy",            # Configures search behavior
    "ContextUpdate",             # Updates conversation context
    
    # Result and response classes - these contain agent outputs
    "QueryUnderstanding",        # What the agent understood from user query
    "RetrievalResult",          # Search results from video database
    "ComprehensiveExplanation", # Detailed content explanations
    "SynthesizedResponse",      # Final polished response
    "ContextAnalysis",          # Analysis of conversation state
    
    # Data structure classes - these represent pieces of information
    "Intent",                   # User's intention (search, explain, etc.)
    "Entity",                   # Named entities in queries (people, objects)
    "SearchResult",            # Individual search result
    "SessionContext",          # Complete conversation session state
    "ConversationTurn",        # Single user-assistant exchange
    "UserPreferences",         # User's saved preferences
    "ResponseComponent",       # Part of a larger response
    "MediaReference",          # Reference to video/audio/image content
    "SceneDescription",        # Description of video scene
    "ObjectExplanation",       # Explanation of objects in video
    "ActionAnalysis"           # Analysis of actions/activities
]

def create_conversation_flow_config(
    understanding_depth="medium",
    search_mode="hybrid", 
    explanation_detail="medium",
    context_memory_length=10,
    response_style="conversational"
):
    """
    Create a configuration that controls how the conversation agents work together.
    
    Think of this as setting the personality and behavior of your AI conversation system.
    Just like how you might adjust the settings on your phone to match your preferences,
    this function lets you tune how the agents understand queries, search for content,
    explain things, and respond to users.
    
    Args:
        understanding_depth: How deeply to analyze user queries
            - "simple": Basic intent and keyword extraction
            - "medium": Include entities and some context analysis  
            - "deep": Full semantic analysis with ambiguity resolution
        search_mode: How to search for relevant video content
            - "keyword": Traditional keyword-based search
            - "semantic": AI-powered meaning-based search
            - "hybrid": Combine both keyword and semantic approaches
        explanation_detail: How much detail to include in explanations
            - "brief": Short, focused explanations
            - "medium": Balanced detail with context
            - "detailed": Comprehensive explanations with background
        context_memory_length: How many previous conversation turns to remember
        response_style: The tone and style of responses
            - "conversational": Friendly, natural language
            - "formal": Professional, structured responses
            - "technical": Include technical details and terminology
            
    Returns:
        dict: Complete configuration for conversation flow
    """
    # Configure query understanding based on depth setting
    understanding_config = {
        "simple": {"extract_entities": True, "expand_queries": False, "resolve_ambiguity": False},
        "medium": {"extract_entities": True, "expand_queries": True, "resolve_ambiguity": True},
        "deep": {"extract_entities": True, "expand_queries": True, "resolve_ambiguity": True, "analyze_context": True}
    }
    
    # Configure search strategy based on mode setting
    search_config = {
        "keyword": {"use_text_search": True, "use_semantic_search": False, "text_weight": 1.0},
        "semantic": {"use_text_search": False, "use_semantic_search": True, "semantic_weight": 1.0},
        "hybrid": {"use_text_search": True, "use_semantic_search": True, "text_weight": 0.4, "semantic_weight": 0.6}
    }
    
    # Configure explanation detail level
    explanation_config = {
        "brief": {"include_background": False, "detail_level": "brief", "max_length": 200},
        "medium": {"include_background": True, "detail_level": "medium", "max_length": 500},
        "detailed": {"include_background": True, "detail_level": "detailed", "max_length": 1000}
    }
    
    return {
        "query_understanding": understanding_config.get(understanding_depth, understanding_config["medium"]),
        "search_strategy": search_config.get(search_mode, search_config["hybrid"]),
        "explanation_settings": explanation_config.get(explanation_detail, explanation_config["medium"]),
        "context_management": {
            "memory_length": context_memory_length,
            "track_entities": True,
            "maintain_topic_coherence": True,
            "detect_context_shifts": True
        },
        "response_synthesis": {
            "style": response_style,
            "include_media": True,
            "provide_followup_suggestions": True,
            "cite_sources": True
        }
    }

def analyze_conversation_quality(conversation_turns):
    """
    Analyze the quality of a conversation to understand how well the agents performed.
    
    This function acts like a conversation coach, looking at how well the AI system
    understood user queries, found relevant content, and provided helpful responses.
    It's similar to how a teacher might review a student's essay to identify strengths
    and areas for improvement.
    
    Args:
        conversation_turns: List of conversation exchanges to analyze
        
    Returns:
        dict: Detailed analysis of conversation quality with recommendations
    """
    if not conversation_turns:
        return {"error": "No conversation turns to analyze"}
    
    analysis = {
        "overall_quality": "analyzing",
        "understanding_accuracy": 0,
        "search_relevance": 0, 
        "explanation_clarity": 0,
        "context_continuity": 0,
        "user_satisfaction": 0,
        "strengths": [],
        "improvement_areas": [],
        "recommendations": []
    }
    
    total_turns = len(conversation_turns)
    successful_understanding = 0
    relevant_results = 0
    clear_explanations = 0
    maintained_context = 0
    
    # Analyze each conversation turn like examining individual test answers
    for i, turn in enumerate(conversation_turns):
        # Check if query was understood correctly
        understanding = turn.get("query_understanding", {})
        if understanding.get("confidence_score", 0) > 0.7:
            successful_understanding += 1
        
        # Check if search results were relevant  
        retrieval = turn.get("retrieval_results", {})
        if retrieval.get("avg_relevance_score", 0) > 0.6:
            relevant_results += 1
            
        # Check if explanations were clear and helpful
        explanation = turn.get("explanation_results", {})
        if explanation.get("clarity_score", 0) > 0.7:
            clear_explanations += 1
            
        # Check if context was maintained from previous turns
        if i > 0:  # Can only check context continuity after first turn
            context_info = turn.get("context_updates", {})
            if context_info.get("context_coherence", 0) > 0.6:
                maintained_context += 1
    
    # Calculate quality scores as percentages
    analysis["understanding_accuracy"] = (successful_understanding / total_turns) * 100
    analysis["search_relevance"] = (relevant_results / total_turns) * 100  
    analysis["explanation_clarity"] = (clear_explanations / total_turns) * 100
    
    # Context continuity only applies to turns after the first
    context_turns = max(total_turns - 1, 1)
    analysis["context_continuity"] = (maintained_context / context_turns) * 100
    
    # Calculate overall quality as weighted average
    analysis["user_satisfaction"] = (
        analysis["understanding_accuracy"] * 0.3 +
        analysis["search_relevance"] * 0.3 + 
        analysis["explanation_clarity"] * 0.2 +
        analysis["context_continuity"] * 0.2
    )
    
    # Determine overall quality category
    if analysis["user_satisfaction"] >= 85:
        analysis["overall_quality"] = "excellent"
        analysis["strengths"].append("High performance across all conversation aspects")
    elif analysis["user_satisfaction"] >= 70:
        analysis["overall_quality"] = "good"
        analysis["strengths"].append("Solid performance with room for optimization")
    elif analysis["user_satisfaction"] >= 55:
        analysis["overall_quality"] = "fair"
        analysis["improvement_areas"].append("Several areas need attention")
    else:
        analysis["overall_quality"] = "needs_improvement"
        analysis["improvement_areas"].append("Significant improvements needed")
    
    # Provide specific recommendations based on weak areas
    if analysis["understanding_accuracy"] < 70:
        analysis["recommendations"].append("Improve query understanding by expanding entity recognition and intent classification training")
    
    if analysis["search_relevance"] < 70:
        analysis["recommendations"].append("Enhance search algorithms and consider adjusting relevance scoring weights")
        
    if analysis["explanation_clarity"] < 70:
        analysis["recommendations"].append("Refine explanation generation to be more clear and user-friendly")
        
    if analysis["context_continuity"] < 70:
        analysis["recommendations"].append("Strengthen context management to better maintain conversation flow")
    
    return analysis

def create_user_persona_config(
    expertise_level="general",
    preferred_response_length="medium",
    interests=None,
    interaction_style="friendly"
):
    """
    Create a configuration that adapts the conversation system to different types of users.
    
    Just like how a good teacher adapts their teaching style to different students,
    this function helps the AI system adjust its behavior to match different user preferences
    and expertise levels. This personalization makes conversations more natural and effective.
    
    Args:
        expertise_level: User's domain expertise
            - "beginner": New to the subject, needs basic explanations
            - "general": Average knowledge, balanced explanations  
            - "expert": Advanced knowledge, can handle technical details
        preferred_response_length: How long responses should be
            - "brief": Quick, concise answers
            - "medium": Balanced detail and brevity
            - "detailed": Comprehensive, thorough responses
        interests: List of topics the user is particularly interested in
        interaction_style: How the AI should communicate
            - "friendly": Warm, casual conversation
            - "professional": Formal, business-like interaction
            - "educational": Teaching-focused, explanation-heavy
            
    Returns:
        dict: User persona configuration for conversation agents
    """
    if interests is None:
        interests = []
    
    # Configure explanation detail based on expertise level
    expertise_configs = {
        "beginner": {
            "explanation_style": "simple",
            "include_background": True,
            "use_analogies": True,
            "define_technical_terms": True,
            "confidence_threshold": 0.8  # Be more certain before answering
        },
        "general": {
            "explanation_style": "balanced", 
            "include_background": True,
            "use_analogies": False,
            "define_technical_terms": False,
            "confidence_threshold": 0.6
        },
        "expert": {
            "explanation_style": "technical",
            "include_background": False,
            "use_analogies": False, 
            "define_technical_terms": False,
            "confidence_threshold": 0.4  # Can provide less certain answers
        }
    }
    
    # Configure response length preferences
    length_configs = {
        "brief": {"max_response_words": 100, "prioritize_key_points": True},
        "medium": {"max_response_words": 300, "prioritize_key_points": False},
        "detailed": {"max_response_words": 600, "prioritize_key_points": False}
    }
    
    # Configure interaction style
    style_configs = {
        "friendly": {
            "tone": "conversational",
            "use_contractions": True,
            "include_encouragement": True,
            "formality_level": "casual"
        },
        "professional": {
            "tone": "formal",
            "use_contractions": False,
            "include_encouragement": False,
            "formality_level": "business"
        },
        "educational": {
            "tone": "instructional",
            "use_contractions": False,
            "include_encouragement": True,
            "formality_level": "academic"
        }
    }
    
    return {
        "expertise_settings": expertise_configs.get(expertise_level, expertise_configs["general"]),
        "response_length": length_configs.get(preferred_response_length, length_configs["medium"]),
        "interaction_style": style_configs.get(interaction_style, style_configs["friendly"]),
        "personalization": {
            "user_interests": interests,
            "adapt_to_feedback": True,
            "remember_preferences": True,
            "learn_from_interactions": True
        }
    }

# Constants that define different conversation modes
class ConversationModes:
    """
    Constants defining different conversation modes for different use cases.
    
    These modes are like different settings on a camera - each optimized for
    a specific situation to get the best results.
    """
    QUICK_CHAT = "quick_chat"           # Fast responses, minimal processing
    DEEP_EXPLORATION = "deep_exploration"  # Thorough analysis, detailed explanations
    EDUCATIONAL = "educational"          # Teaching-focused, step-by-step explanations
    RESEARCH = "research"               # Comprehensive search, multiple perspectives
    CASUAL_BROWSING = "casual_browsing"  # Relaxed exploration, varied content

# Performance targets for conversation quality
class QualityTargets:
    """
    Target performance metrics for conversation quality assessment.
    
    These targets help you understand whether your conversation system is
    performing at the level needed for a good user experience.
    """
    MIN_UNDERSTANDING_ACCURACY = 75    # Minimum % of queries understood correctly
    MIN_SEARCH_RELEVANCE = 70         # Minimum % relevance for search results
    MIN_EXPLANATION_CLARITY = 80      # Minimum clarity score for explanations
    MIN_CONTEXT_CONTINUITY = 65       # Minimum context maintenance score
    TARGET_RESPONSE_TIME = 3          # Target response time in seconds
    MAX_RESPONSE_TIME = 10            # Maximum acceptable response time

# Default configurations for common scenarios
CONVERSATION_PRESETS = {
    "customer_support": create_conversation_flow_config(
        understanding_depth="deep",
        search_mode="hybrid", 
        explanation_detail="detailed",
        context_memory_length=15,
        response_style="professional"
    ),
    "educational_content": create_conversation_flow_config(
        understanding_depth="medium",
        search_mode="semantic",
        explanation_detail="detailed", 
        context_memory_length=8,
        response_style="conversational"
    ),
    "quick_answers": create_conversation_flow_config(
        understanding_depth="simple",
        search_mode="keyword",
        explanation_detail="brief",
        context_memory_length=5,
        response_style="conversational"
    )
}