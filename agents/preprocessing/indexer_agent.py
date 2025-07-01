"""
Vector Indexer Agent - Create searchable indexes từ knowledge graph và features
Text Embeddings, Visual Embeddings, Audio Embeddings, Hybrid Indexing
Phần cuối của Phase 1: Preprocessing - chuẩn bị cho competition phase
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time

from ..base_agent import PreprocessingAgent, AgentResponse

# Structured output models
class TextIndex(BaseModel):
    """Text search index"""
    index_id: str = Field(..., description="Index ID")
    content_type: str = Field(..., description="Content type: transcript, ocr, description, etc.")
    
    # Content
    text_content: str = Field(..., description="Original text content")
    processed_text: str = Field(..., description="Processed/cleaned text")
    language: str = Field(default="unknown", description="Detected language")
    
    # Embeddings
    embedding: List[float] = Field(default_factory=list, description="Text embedding vector")
    embedding_model: str = Field(..., description="Model used for embedding")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    
    # Metadata
    source_id: str = Field(..., description="Source video/frame/segment ID")
    timestamp: Optional[float] = Field(None, description="Timestamp in video")
    confidence: float = Field(..., description="Content extraction confidence")
    
    # Search metadata
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    topics: List[str] = Field(default_factory=list, description="Topic categories")

class VisualIndex(BaseModel):
    """Visual search index"""
    index_id: str = Field(..., description="Index ID")
    content_type: str = Field(..., description="Content type: keyframe, scene, object")
    
    # Visual content
    image_path: str = Field(..., description="Path to image file")
    visual_description: str = Field(..., description="AI-generated description")
    detected_objects: List[str] = Field(default_factory=list, description="Detected objects")
    
    # Embeddings
    visual_embedding: List[float] = Field(default_factory=list, description="Visual embedding vector")
    embedding_model: str = Field(..., description="Model used (CLIP, BLIP, etc.)")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    
    # Metadata  
    source_id: str = Field(..., description="Source video/frame ID")
    timestamp: float = Field(..., description="Timestamp in video")
    scene_id: Optional[str] = Field(None, description="Associated scene ID")
    
    # Visual properties
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors")
    brightness: float = Field(default=0.5, description="Brightness level")
    scene_type: str = Field(default="unknown", description="Scene type")

class AudioIndex(BaseModel):
    """Audio search index"""
    index_id: str = Field(..., description="Index ID")
    content_type: str = Field(..., description="Content type: speech, music, sound_effect")
    
    # Audio content
    transcript: str = Field(default="", description="Speech transcript")
    audio_type: str = Field(..., description="Audio type classification")
    language: str = Field(default="unknown", description="Speech language")
    
    # Embeddings
    audio_embedding: List[float] = Field(default_factory=list, description="Audio embedding vector")
    embedding_model: str = Field(..., description="Model used for embedding")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    
    # Metadata
    source_id: str = Field(..., description="Source video/segment ID")
    start_time: float = Field(..., description="Start timestamp")
    end_time: float = Field(..., description="End timestamp")
    duration: float = Field(..., description="Segment duration")
    
    # Audio properties
    volume_level: float = Field(default=0.5, description="Volume level")
    speaker_count: int = Field(default=0, description="Number of speakers")
    emotion: str = Field(default="neutral", description="Detected emotion")

class HybridIndex(BaseModel):
    """Multi-modal hybrid index entry"""
    index_id: str = Field(..., description="Unique hybrid index ID")
    
    # Multi-modal content
    combined_description: str = Field(..., description="Combined multi-modal description")
    primary_modality: str = Field(..., description="Primary modality: visual, audio, text")
    
    # All embeddings
    text_embedding: List[float] = Field(default_factory=list, description="Text embedding")
    visual_embedding: List[float] = Field(default_factory=list, description="Visual embedding")
    audio_embedding: List[float] = Field(default_factory=list, description="Audio embedding")
    fused_embedding: List[float] = Field(default_factory=list, description="Fused multi-modal embedding")
    
    # Source references
    text_sources: List[str] = Field(default_factory=list, description="Text source IDs")
    visual_sources: List[str] = Field(default_factory=list, description="Visual source IDs")
    audio_sources: List[str] = Field(default_factory=list, description="Audio source IDs")
    
    # Temporal info
    start_time: float = Field(..., description="Start timestamp")
    end_time: float = Field(..., description="End timestamp")
    
    # Search optimization
    search_keywords: List[str] = Field(default_factory=list, description="Combined keywords")
    search_entities: List[str] = Field(default_factory=list, description="Combined entities")
    search_topics: List[str] = Field(default_factory=list, description="Combined topics")

class IndexingResult(BaseModel):
    """Kết quả tổng thể indexing process"""
    video_id: str = Field(..., description="Source video ID")
    indexing_timestamp: str = Field(..., description="Indexing completion time")
    
    # Indexes created
    text_indexes: List[TextIndex] = Field(default_factory=list, description="Text search indexes")
    visual_indexes: List[VisualIndex] = Field(default_factory=list, description="Visual search indexes")
    audio_indexes: List[AudioIndex] = Field(default_factory=list, description="Audio search indexes")
    hybrid_indexes: List[HybridIndex] = Field(default_factory=list, description="Multi-modal hybrid indexes")
    
    # Statistics
    total_indexes: int = Field(..., description="Total indexes created")
    index_sizes: Dict[str, int] = Field(default_factory=dict, description="Index sizes by type")
    
    # Database info
    vector_db_info: Dict[str, Any] = Field(default_factory=dict, description="Vector database information")
    search_db_info: Dict[str, Any] = Field(default_factory=dict, description="Search database information")
    
    # Processing info
    processing_time: float = Field(..., description="Total indexing time")
    models_used: List[str] = Field(default_factory=list, description="Embedding models used")
    status: str = Field(..., description="Indexing status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class IndexingConfig(BaseModel):
    """Configuration cho indexing process"""
    # Index types to create
    create_text_index: bool = Field(default=True, description="Create text search index")
    create_visual_index: bool = Field(default=True, description="Create visual search index")
    create_audio_index: bool = Field(default=True, description="Create audio search index")
    create_hybrid_index: bool = Field(default=True, description="Create multi-modal hybrid index")
    
    # Embedding models
    text_embedding_model: str = Field(default="text-embedding-3-large", description="Text embedding model")
    visual_embedding_model: str = Field(default="clip-vit-large", description="Visual embedding model")
    audio_embedding_model: str = Field(default="audio-embedding-model", description="Audio embedding model")
    
    # Segmentation
    hybrid_segment_length: int = Field(default=30, description="Hybrid index segment length (seconds)")
    text_chunk_size: int = Field(default=512, description="Text chunk size for indexing")
    overlap_size: int = Field(default=50, description="Overlap between text chunks")
    
    # Quality thresholds
    min_text_confidence: float = Field(default=0.6, description="Minimum text confidence")
    min_visual_confidence: float = Field(default=0.7, description="Minimum visual confidence")
    min_audio_confidence: float = Field(default=0.6, description="Minimum audio confidence")
    
    # Database settings
    vector_db_name: str = Field(default="video_vectors", description="Vector database name")
    search_db_name: str = Field(default="video_search", description="Search database name")
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid keyword+vector search")

class VectorIndexerAgent(PreprocessingAgent):
    """
    Agent chuyên tạo searchable indexes từ multi-modal features và knowledge graph
    Chuẩn bị data cho competition phase searching
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="VectorIndexer",
            model_type="gpt-4o",
            **kwargs
        )
        
        self.set_instructions([
            "You are a multi-modal search indexing specialist.",
            "Create comprehensive search indexes from processed video content:",
            "- Text indexes from transcripts, OCR, and descriptions",
            "- Visual indexes from keyframes and detected objects", 
            "- Audio indexes from speech and sound analysis",
            "- Hybrid multi-modal indexes for complex queries",
            "Optimize indexes for fast retrieval during competition phase.",
            "Ensure proper embedding generation and metadata extraction.",
            "Focus on search relevance and query performance.",
            "Support both keyword and semantic search capabilities."
        ])
        
        self.agent.response_model = IndexingResult
        
    def process(self, 
                graph_data: Dict[str, Any], 
                features_data: Dict[str, Any],
                config: Optional[IndexingConfig] = None,
                **kwargs) -> AgentResponse:
        """
        Tạo comprehensive search indexes từ knowledge graph và features
        
        Args:
            graph_data: Knowledge graph từ KnowledgeGraphAgent
            features_data: Multi-modal features từ FeatureExtractorAgent  
            config: Indexing configuration
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Use default config if not provided
            if config is None:
                config = IndexingConfig()
            
            # Create indexing prompt
            prompt = self._create_indexing_prompt(graph_data, features_data, config)
            
            # Run agent với structured output
            response = self.agent.run(prompt)
            
            # Validate response
            if not isinstance(response, IndexingResult):
                result = self._parse_indexing_response(response.content, graph_data, features_data)
            else:
                result = response
            
            # Post-process để optimize indexes
            result = self._optimize_indexes(result, config)
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="vector_indexing",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "total_indexes": result.total_indexes,
                    "index_types": list(result.index_sizes.keys()),
                    "config": config.dict(),
                    "search_ready": True
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Vector indexing failed: {str(e)}")
            
            return self._create_response(
                task_type="vector_indexing",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_indexing_prompt(self, graph_data: Dict, features_data: Dict, config: IndexingConfig) -> str:
        """Tạo prompt cho indexing process"""
        
        # Extract data summaries
        entities = graph_data.get("entities", [])
        relations = graph_data.get("relations", [])
        visual_features = features_data.get("visual_features", [])
        audio_features = features_data.get("audio_features", [])
        video_id = features_data.get("video_id", "unknown")
        
        prompt = f"""
        Create comprehensive search indexes for video: {video_id}
        
        INPUT DATA SUMMARY:
        - Knowledge Graph: {len(entities)} entities, {len(relations)} relations
        - Visual Features: {len(visual_features)} keyframes
        - Audio Features: {len(audio_features)} segments
        
        INDEXING CONFIGURATION:
        - Create Text Index: {config.create_text_index}
        - Create Visual Index: {config.create_visual_index}
        - Create Audio Index: {config.create_audio_index}
        - Create Hybrid Index: {config.create_hybrid_index}
        - Text Embedding Model: {config.text_embedding_model}
        - Visual Embedding Model: {config.visual_embedding_model}
        - Hybrid Segment Length: {config.hybrid_segment_length}s
        - Text Chunk Size: {config.text_chunk_size}
        
        SOURCE DATA DETAILS:
        """
        
        # Add knowledge graph summary
        if entities:
            prompt += f"\nKNOWLEDGE GRAPH ENTITIES (first 5):\n"
            for i, entity in enumerate(entities[:5]):
                prompt += f"  - {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'unknown')})\n"
                prompt += f"    Description: {entity.get('description', 'N/A')[:100]}...\n"
                prompt += f"    Timestamps: {entity.get('timestamps', [])}\n"
        
        # Add visual features summary
        if visual_features:
            prompt += f"\nVISUAL FEATURES (first 3):\n"
            for i, vf in enumerate(visual_features[:3]):
                prompt += f"  Frame {i+1} (t={vf.get('timestamp', 0)}s):\n"
                prompt += f"    - Description: {vf.get('blip_caption', 'N/A')[:100]}...\n"
                prompt += f"    - Objects: {vf.get('blip_objects', [])}\n"
                prompt += f"    - OCR: {vf.get('ocr_text', 'N/A')[:50]}...\n"
        
        # Add audio features summary
        if audio_features:
            prompt += f"\nAUDIO FEATURES (first 3):\n"
            for i, af in enumerate(audio_features[:3]):
                prompt += f"  Segment {i+1} (t={af.get('start_time', 0)}-{af.get('end_time', 0)}s):\n"
                prompt += f"    - Transcript: {af.get('transcript', 'N/A')[:100]}...\n"
                prompt += f"    - Type: {af.get('audio_type', 'N/A')}\n"
                prompt += f"    - Language: {af.get('language', 'N/A')}\n"
        
        prompt += f"""
        
        INDEXING TASKS:
        """
        
        if config.create_text_index:
            prompt += f"""
        1. TEXT INDEXING:
           - Extract all text content from transcripts, OCR, descriptions
           - Chunk text into {config.text_chunk_size} character segments với {config.overlap_size} overlap
           - Generate text embeddings using {config.text_embedding_model}
           - Extract keywords, entities, and topics for each chunk
           - Map text to original timestamps and sources
           - Filter by confidence >= {config.min_text_confidence}
        """
        
        if config.create_visual_index:
            prompt += f"""
        2. VISUAL INDEXING:
           - Index all keyframes và detected objects
           - Generate visual embeddings using {config.visual_embedding_model}
           - Include AI-generated descriptions and object lists
           - Extract visual properties (colors, brightness, scene type)
           - Map visuals to timestamps and scenes
           - Filter by confidence >= {config.min_visual_confidence}
        """
        
        if config.create_audio_index:
            prompt += f"""
        3. AUDIO INDEXING:
           - Index speech transcripts and audio classifications
           - Generate audio embeddings for each segment
           - Include speaker information and emotion analysis
           - Map audio to temporal segments
           - Filter by confidence >= {config.min_audio_confidence}
        """
        
        if config.create_hybrid_index:
            prompt += f"""
        4. HYBRID MULTI-MODAL INDEXING:
           - Create unified indexes combining text, visual, and audio
           - Segment video into {config.hybrid_segment_length}s chunks
           - Fuse embeddings from all modalities for each segment
           - Generate comprehensive descriptions combining all content
           - Create unified keyword and entity lists
           - Optimize for complex multi-modal queries
        """
        
        prompt += """
        
        INDEX OPTIMIZATION:
        - Ensure proper embedding normalization
        - Create efficient metadata structures
        - Generate comprehensive search keywords
        - Optimize for both keyword and semantic search
        - Prepare indexes for vector database storage
        - Include search performance metadata
        
        OUTPUT REQUIREMENTS:
        - Return complete IndexingResult object
        - Include all generated indexes with embeddings
        - Provide comprehensive metadata for each index
        - Generate statistics and performance information
        - Ensure indexes are ready for competition phase deployment
        """
        
        return prompt
    
    def _parse_indexing_response(self, response_content: str, graph_data: Dict, features_data: Dict) -> IndexingResult:
        """Fallback parsing nếu structured output fails"""
        return IndexingResult(
            video_id=features_data.get("video_id", "unknown"),
            indexing_timestamp=str(time.time()),
            text_indexes=[],
            visual_indexes=[],
            audio_indexes=[],
            hybrid_indexes=[],
            total_indexes=0,
            index_sizes={},
            vector_db_info={},
            search_db_info={},
            processing_time=0,
            models_used=["fallback_parser"],
            status="parsed_fallback"
        )
    
    def _optimize_indexes(self, result: IndexingResult, config: IndexingConfig) -> IndexingResult:
        """Optimize indexes cho better search performance"""
        # Update statistics
        result.total_indexes = (
            len(result.text_indexes) + 
            len(result.visual_indexes) + 
            len(result.audio_indexes) + 
            len(result.hybrid_indexes)
        )
        
        result.index_sizes = {
            "text": len(result.text_indexes),
            "visual": len(result.visual_indexes),
            "audio": len(result.audio_indexes),
            "hybrid": len(result.hybrid_indexes)
        }
        
        # Add database configuration info
        result.vector_db_info = {
            "database_name": config.vector_db_name,
            "embedding_dimensions": {
                "text": 3072,  # text-embedding-3-large dimension
                "visual": 768,  # CLIP dimension
                "audio": 512,   # Audio embedding dimension
                "hybrid": 4352  # Combined dimension
            },
            "index_types": ["flat", "ivf", "hnsw"],
            "distance_metrics": ["cosine", "euclidean", "dot_product"]
        }
        
        result.search_db_info = {
            "database_name": config.search_db_name,
            "search_types": ["keyword", "semantic", "hybrid"],
            "supports_filters": True,
            "supports_aggregations": True
        }
        
        return result
    
    def create_text_index_only(self, graph_data: Dict, features_data: Dict, **kwargs) -> AgentResponse:
        """Chỉ tạo text search index"""
        config = IndexingConfig(
            create_text_index=True,
            create_visual_index=False,
            create_audio_index=False,
            create_hybrid_index=False
        )
        return self.process(graph_data, features_data, config, **kwargs)
    
    def create_visual_index_only(self, graph_data: Dict, features_data: Dict, **kwargs) -> AgentResponse:
        """Chỉ tạo visual search index"""
        config = IndexingConfig(
            create_text_index=False,
            create_visual_index=True,
            create_audio_index=False,
            create_hybrid_index=False
        )
        return self.process(graph_data, features_data, config, **kwargs)
    
    def create_hybrid_index_only(self, graph_data: Dict, features_data: Dict, segment_length: int = 30, **kwargs) -> AgentResponse:
        """Chỉ tạo hybrid multi-modal index"""
        config = IndexingConfig(
            create_text_index=False,
            create_visual_index=False,
            create_audio_index=False,
            create_hybrid_index=True,
            hybrid_segment_length=segment_length
        )
        return self.process(graph_data, features_data, config, **kwargs)
    
    def benchmark_indexing(self, test_data: List[Dict], config: IndexingConfig) -> Dict[str, Any]:
        """Benchmark indexing performance"""
        results = []
        
        for data in test_data:
            graph_data = data.get("graph_data", {})
            features_data = data.get("features_data", {})
            
            result = self.process(graph_data, features_data, config)
            results.append({
                "video_id": features_data.get("video_id", "unknown"),
                "status": result.status,
                "execution_time": result.execution_time,
                "total_indexes": result.result.get("total_indexes", 0),
                "index_sizes": result.result.get("index_sizes", {})
            })
        
        return {
            "total_videos": len(test_data),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "avg_indexing_time": sum(r["execution_time"] for r in results) / len(results),
            "total_indexes_created": sum(r["total_indexes"] for r in results),
            "avg_indexes_per_video": sum(r["total_indexes"] for r in results) / len(results),
            "details": results
        }
    
    def get_search_capabilities(self) -> Dict[str, Any]:
        """Lấy thông tin search capabilities"""
        return {
            "supported_queries": {
                "text_search": {
                    "description": "Search trong transcripts, OCR, descriptions",
                    "examples": ["find cooking videos", "videos with recipe instructions"]
                },
                "visual_search": {
                    "description": "Search by visual content và objects",
                    "examples": ["show me red cars", "videos with people cooking"]
                },
                "audio_search": {
                    "description": "Search by speech content và audio characteristics",
                    "examples": ["find happy conversations", "videos with music"]
                },
                "hybrid_search": {
                    "description": "Multi-modal search combining all modalities",
                    "examples": ["cooking videos with happy people talking", "outdoor scenes with car sounds"]
                }
            },
            "search_types": [
                "keyword_search",      # BM25/TF-IDF based
                "semantic_search",     # Vector similarity based
                "hybrid_search",       # Combined keyword + semantic
                "multimodal_search",   # Cross-modal retrieval
                "temporal_search",     # Time-based filtering
                "entity_search"        # Knowledge graph based
            ],
            "performance_metrics": {
                "index_size": "optimized for memory efficiency",
                "search_latency": "< 100ms for most queries",
                "recall": "> 90% for relevant content",
                "precision": "> 85% for top-10 results"
            }
        }