"""
AI Challenge 2.0 - Agents Package
Multi-Modal Conversational Video AI System

This package contains all agents for the video AI system:
- Base agent classes and utilities
- Preprocessing agents for Phase 1 (offline video processing)
- Conversational agents for Phase 2 (real-time chat)
- Orchestrators for coordinating agent workflows
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Challenge Team"
__description__ = "Multi-agent system for conversational video AI"

# Import base classes
from .base_agent import (
    BaseAIAgent,
    PreprocessingAgent, 
    ConversationalAgent,
    AgentResponse
)

# Import preprocessing agents
from .preprocessing import (
    VideoProcessorAgent,
    FeatureExtractorAgent,
    KnowledgeGraphAgent,
    VectorIndexerAgent
)

# Import conversational agents  
from .conversational import (
    QueryUnderstandingAgent,
    VideoRetrievalAgent,
    ContentExplainerAgent,
    ContextManagerAgent,
    ResponseSynthesisAgent
)

# Import orchestrators
from .orchestrator import (
    ConversationOrchestrator,
    PreprocessingOrchestrator
)

# Define what gets imported with "from agents import *"
__all__ = [
    # Base classes
    "BaseAIAgent",
    "PreprocessingAgent", 
    "ConversationalAgent",
    "AgentResponse",
    
    # Preprocessing agents
    "VideoProcessorAgent",
    "FeatureExtractorAgent", 
    "KnowledgeGraphAgent",
    "VectorIndexerAgent",
    
    # Conversational agents
    "QueryUnderstandingAgent",
    "VideoRetrievalAgent",
    "ContentExplainerAgent", 
    "ContextManagerAgent",
    "ResponseSynthesisAgent",
    
    # Orchestrators
    "ConversationOrchestrator",
    "PreprocessingOrchestrator"
]

# Package-level utilities
def get_agent_info():
    """Get information about all available agents"""
    return {
        "package_version": __version__,
        "total_agents": len(__all__) - 1,  # Excluding AgentResponse
        "preprocessing_agents": [
            "VideoProcessorAgent",
            "FeatureExtractorAgent", 
            "KnowledgeGraphAgent",
            "VectorIndexerAgent"
        ],
        "conversational_agents": [
            "QueryUnderstandingAgent",
            "VideoRetrievalAgent",
            "ContentExplainerAgent", 
            "ContextManagerAgent", 
            "ResponseSynthesisAgent"
        ],
        "orchestrators": [
            "ConversationOrchestrator",
            "PreprocessingOrchestrator"
        ]
    }

def create_preprocessing_pipeline():
    """
    Factory function to create a complete preprocessing pipeline
    
    Returns:
        PreprocessingOrchestrator: Configured preprocessing orchestrator
    """
    return PreprocessingOrchestrator()

def create_conversation_system():
    """
    Factory function to create a complete conversation system
    
    Returns:
        ConversationOrchestrator: Configured conversation orchestrator
    """
    return ConversationOrchestrator()

# Setup package-level logging
import logging

def setup_agent_logging(level=logging.INFO, log_file=None):
    """
    Setup logging for all agents
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    logger = logging.getLogger("agents")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize default logging
setup_agent_logging()

# Package constants
class AgentTypes:
    """Constants for agent types"""
    PREPROCESSING = "preprocessing"
    CONVERSATIONAL = "conversational"
    ORCHESTRATOR = "orchestrator"

class ProcessingPhases:
    """Constants for processing phases"""
    PHASE_1_PREPROCESSING = "phase_1_preprocessing"
    PHASE_2_COMPETITION = "phase_2_competition"

# Validation function
def validate_agent_dependencies():
    """
    Validate that all required dependencies are available
    
    Returns:
        dict: Validation results
    """
    try:
        import agno
        import pydantic
        import time
        import logging
        
        return {
            "status": "success",
            "agno_available": True,
            "pydantic_available": True,
            "all_dependencies_met": True
        }
    except ImportError as e:
        return {
            "status": "error", 
            "error": str(e),
            "agno_available": False,
            "pydantic_available": False,
            "all_dependencies_met": False
        }