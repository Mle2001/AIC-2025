"""
Preprocessing Agents Package - Phase 1 of AI Challenge System

This subpackage contains all agents responsible for Phase 1 preprocessing:
- Converting raw video files into structured, searchable data
- Extracting multi-modal features (visual, audio, text)
- Building knowledge graphs from extracted content
- Creating vector indexes for fast search and retrieval

The preprocessing phase runs offline before the competition to prepare
all video content for real-time search and conversation during Phase 2.
"""

# Import all preprocessing agents for easy access
from .video_processor_agent import (
    VideoProcessorAgent,
    VideoProcessingResult,
    SceneDetection,
    KeyFrame,
    ShotBoundary
)

from .feature_extractor_agent import (
    FeatureExtractorAgent,
    FeatureExtractionConfig,
    MultiModalFeatures,
    VisualFeature,
    AudioFeature
)

from .knowledge_graph_agent import (
    KnowledgeGraphAgent,
    GraphBuildingConfig,
    KnowledgeGraph,
    Entity,
    Relation,
    GraphSchema
)

from .indexer_agent import (
    VectorIndexerAgent,
    IndexingConfig,
    IndexingResult,
    TextIndex,
    VisualIndex,
    AudioIndex,
    HybridIndex
)

# Export key classes and configurations
__all__ = [
    # Main agent classes
    "VideoProcessorAgent",
    "FeatureExtractorAgent", 
    "KnowledgeGraphAgent",
    "VectorIndexerAgent",
    
    # Configuration classes
    "FeatureExtractionConfig",
    "GraphBuildingConfig", 
    "IndexingConfig",
    
    # Result classes
    "VideoProcessingResult",
    "MultiModalFeatures",
    "KnowledgeGraph",
    "IndexingResult",
    
    # Data model classes
    "SceneDetection",
    "KeyFrame", 
    "ShotBoundary",
    "VisualFeature",
    "AudioFeature",
    "Entity",
    "Relation",
    "GraphSchema",
    "TextIndex",
    "VisualIndex", 
    "AudioIndex",
    "HybridIndex"
]

# Preprocessing pipeline utilities
def create_preprocessing_pipeline_config(
    extract_visual=True,
    extract_audio=True, 
    extract_ocr=True,
    build_knowledge_graph=True,
    create_vector_indexes=True,
    use_gpu=True
):
    """
    Create a comprehensive configuration for the preprocessing pipeline.
    
    This function helps you set up all the configurations needed for processing
    videos through the complete pipeline. Think of it as a recipe that tells
    each agent exactly what to do and how to do it.
    
    Args:
        extract_visual: Whether to extract visual features like CLIP embeddings
        extract_audio: Whether to extract audio features and transcripts  
        extract_ocr: Whether to extract text from video frames
        build_knowledge_graph: Whether to build entity relationships
        create_vector_indexes: Whether to create searchable indexes
        use_gpu: Whether to use GPU acceleration when available
        
    Returns:
        dict: Complete configuration for all preprocessing agents
    """
    return {
        "feature_extraction": FeatureExtractionConfig(
            extract_visual=extract_visual,
            extract_audio=extract_audio,
            extract_ocr=extract_ocr,
            use_clip=extract_visual,
            use_blip=extract_visual,
            use_whisper=extract_audio
        ),
        "graph_building": GraphBuildingConfig(
            extract_persons=build_knowledge_graph,
            extract_objects=build_knowledge_graph,
            extract_locations=build_knowledge_graph,
            extract_concepts=build_knowledge_graph,
            extract_actions=build_knowledge_graph,
            merge_similar_entities=True
        ),
        "indexing": IndexingConfig(
            create_text_index=create_vector_indexes,
            create_visual_index=create_vector_indexes,
            create_audio_index=create_vector_indexes,
            create_hybrid_index=create_vector_indexes,
            enable_hybrid_search=True
        ),
        "performance": {
            "use_gpu": use_gpu,
            "parallel_workers": 4 if use_gpu else 2,
            "batch_size": 8 if use_gpu else 4
        }
    }

def estimate_preprocessing_resources(video_count, avg_video_length_minutes=10):
    """
    Estimate the computational resources needed for preprocessing a batch of videos.
    
    This helps you plan your preprocessing run by understanding what resources
    you'll need and how long it might take. Think of it as checking if you have
    enough ingredients before starting to cook a large meal.
    
    Args:
        video_count: Number of videos to process
        avg_video_length_minutes: Average length of videos in minutes
        
    Returns:
        dict: Resource estimates including time, memory, and storage
    """
    # Base estimates per minute of video (these are rough estimates)
    base_processing_time_per_minute = 30  # seconds
    base_memory_per_video_gb = 2
    base_storage_per_video_gb = 0.5
    
    total_video_minutes = video_count * avg_video_length_minutes
    
    return {
        "estimated_processing_time": {
            "total_hours": round((total_video_minutes * base_processing_time_per_minute) / 3600, 1),
            "with_4_workers_hours": round((total_video_minutes * base_processing_time_per_minute) / (3600 * 4), 1),
            "with_8_workers_hours": round((total_video_minutes * base_processing_time_per_minute) / (3600 * 8), 1)
        },
        "memory_requirements": {
            "peak_memory_gb": base_memory_per_video_gb * min(video_count, 4),  # Assume max 4 parallel
            "recommended_memory_gb": base_memory_per_video_gb * min(video_count, 4) * 1.5  # 50% buffer
        },
        "storage_requirements": {
            "processed_data_gb": round(video_count * base_storage_per_video_gb, 1),
            "with_backup_gb": round(video_count * base_storage_per_video_gb * 2, 1),
            "temp_storage_gb": round(video_count * base_storage_per_video_gb * 0.5, 1)
        },
        "recommendations": [
            f"Process in batches of {min(video_count, 50)} videos for better memory management",
            "Use SSD storage for faster I/O operations", 
            "Monitor GPU memory if using visual processing",
            "Keep at least 20% free disk space during processing"
        ]
    }

def validate_preprocessing_setup():
    """
    Validate that all required dependencies and resources are available for preprocessing.
    
    This function acts like a pre-flight check before starting your preprocessing pipeline.
    It ensures that all the tools and libraries are properly installed and configured.
    
    Returns:
        dict: Validation results with specific recommendations
    """
    validation_results = {
        "status": "checking",
        "dependencies": {},
        "recommendations": [],
        "critical_issues": [],
        "warnings": []
    }
    
    # Check core dependencies
    try:
        import cv2
        validation_results["dependencies"]["opencv"] = {"available": True, "version": cv2.__version__}
    except ImportError:
        validation_results["dependencies"]["opencv"] = {"available": False}
        validation_results["critical_issues"].append("OpenCV not found - required for video processing")
    
    try:
        import whisper
        validation_results["dependencies"]["whisper"] = {"available": True}
    except ImportError:
        validation_results["dependencies"]["whisper"] = {"available": False}
        validation_results["warnings"].append("Whisper not found - audio processing will be limited")
    
    try:
        import transformers
        validation_results["dependencies"]["transformers"] = {"available": True}
    except ImportError:
        validation_results["dependencies"]["transformers"] = {"available": False}
        validation_results["critical_issues"].append("Transformers not found - required for CLIP/BLIP models")
    
    try:
        import easyocr
        validation_results["dependencies"]["easyocr"] = {"available": True}
    except ImportError:
        validation_results["dependencies"]["easyocr"] = {"available": False}
        validation_results["warnings"].append("EasyOCR not found - text extraction will be limited")
    
    # Determine overall status
    if validation_results["critical_issues"]:
        validation_results["status"] = "critical_issues_found"
        validation_results["recommendations"].append("Install missing critical dependencies before proceeding")
    elif validation_results["warnings"]:
        validation_results["status"] = "warnings_found"
        validation_results["recommendations"].append("Consider installing optional dependencies for full functionality")
    else:
        validation_results["status"] = "ready"
        validation_results["recommendations"].append("All dependencies available - ready for preprocessing")
    
    return validation_results

# Constants for preprocessing stages
class PreprocessingStages:
    """
    Constants defining the sequence of preprocessing stages.
    
    These stages represent the logical flow of transforming raw video content
    into search-ready indexes. Each stage builds upon the results of previous stages.
    """
    VIDEO_PROCESSING = "video_processing"      # Extract scenes, keyframes, shots
    FEATURE_EXTRACTION = "feature_extraction"  # Extract multi-modal features  
    KNOWLEDGE_GRAPH = "knowledge_graph"        # Build entity relationships
    VECTOR_INDEXING = "vector_indexing"        # Create searchable indexes
    VALIDATION = "validation"                  # Quality assurance

# Quality thresholds for preprocessing validation
class QualityThresholds:
    """
    Quality thresholds used to validate preprocessing outputs.
    
    These thresholds help ensure that each stage produces high-quality results
    that will support effective search and conversation in Phase 2.
    """
    MIN_SCENES_PER_VIDEO = 3           # Minimum number of scenes to detect
    MIN_KEYFRAMES_PER_VIDEO = 10       # Minimum number of keyframes to extract
    MIN_FEATURE_COMPLETENESS = 0.8     # Minimum feature extraction completeness
    MIN_ENTITY_COUNT = 5               # Minimum entities in knowledge graph
    MIN_INDEX_COVERAGE = 0.9           # Minimum content coverage in indexes

# Default configurations optimized for different use cases
DEFAULT_CONFIGS = {
    "fast_processing": create_preprocessing_pipeline_config(
        extract_visual=True,
        extract_audio=False,  # Skip audio for speed
        extract_ocr=False,    # Skip OCR for speed
        build_knowledge_graph=True,
        create_vector_indexes=True,
        use_gpu=True
    ),
    "high_quality": create_preprocessing_pipeline_config(
        extract_visual=True,
        extract_audio=True,
        extract_ocr=True,
        build_knowledge_graph=True,
        create_vector_indexes=True,
        use_gpu=True
    ),
    "cpu_only": create_preprocessing_pipeline_config(
        extract_visual=True,
        extract_audio=True,
        extract_ocr=True,
        build_knowledge_graph=True,
        create_vector_indexes=True,
        use_gpu=False
    )
}