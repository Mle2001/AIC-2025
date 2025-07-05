"""
Preprocessing Orchestrator - Điều phối preprocessing pipeline trong Phase 1
Coordinate các preprocessing agents để xử lý video từ raw tới search-ready indexes
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
from enum import Enum
from pathlib import Path

from ..base_agent import PreprocessingAgent, AgentResponse
from ..preprocessing.video_processor_agent import VideoProcessorAgent
from ..preprocessing.feature_extractor_agent import FeatureExtractorAgent, FeatureExtractionConfig
from ..preprocessing.knowledge_graph_agent import KnowledgeGraphAgent, GraphBuildingConfig
from ..preprocessing.indexer_agent import VectorIndexerAgent, IndexingConfig

# Structured output models
class ProcessingStage(str, Enum):
    """Preprocessing stages"""
    INITIALIZING = "initializing"
    VIDEO_PROCESSING = "video_processing"
    FEATURE_EXTRACTION = "feature_extraction"
    KNOWLEDGE_GRAPH_BUILDING = "knowledge_graph_building"
    VECTOR_INDEXING = "vector_indexing"
    VALIDATION = "validation"
    COMPLETED = "completed"
    ERROR = "error"

class StageExecution(BaseModel):
    """Individual stage execution result"""
    stage_name: str = Field(..., description="Stage name")
    agent_name: str = Field(..., description="Responsible agent")
    start_time: float = Field(..., description="Stage start time")
    end_time: float = Field(..., description="Stage end time")
    execution_time: float = Field(..., description="Stage execution time")
    status: str = Field(..., description="Stage status")
    result: Dict[str, Any] = Field(default_factory=dict, description="Stage result")
    error_message: Optional[str] = Field(None, description="Error nếu có")
    
    # Quality metrics
    output_quality: Optional[float] = Field(None, description="Output quality score")
    data_completeness: Optional[float] = Field(None, description="Data completeness score")

class ProcessingPipeline(BaseModel):
    """Complete preprocessing pipeline result"""
    pipeline_id: str = Field(..., description="Unique pipeline ID")
    video_path: str = Field(..., description="Input video path")
    video_id: str = Field(..., description="Video ID")
    
    # Pipeline execution
    current_stage: ProcessingStage = Field(..., description="Current processing stage")
    stage_executions: List[StageExecution] = Field(default_factory=list, description="Stage execution results")
    
    # Results from each stage
    video_processing_result: Optional[Dict[str, Any]] = Field(None, description="Video processing results")
    feature_extraction_result: Optional[Dict[str, Any]] = Field(None, description="Feature extraction results")
    knowledge_graph_result: Optional[Dict[str, Any]] = Field(None, description="Knowledge graph results")
    indexing_result: Optional[Dict[str, Any]] = Field(None, description="Vector indexing results")
    
    # Pipeline metadata
    total_processing_time: float = Field(..., description="Total processing time")
    success_rate: float = Field(..., description="Success rate of stages")
    overall_quality_score: float = Field(..., description="Overall quality score")
    
    # Output info
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    index_locations: List[str] = Field(default_factory=list, description="Created index locations")
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Status
    status: str = Field(..., description="Overall pipeline status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class BatchProcessingResult(BaseModel):
    """Batch processing result"""
    batch_id: str = Field(..., description="Batch ID")
    total_videos: int = Field(..., description="Total videos in batch")
    processed_videos: int = Field(..., description="Successfully processed videos")
    failed_videos: int = Field(..., description="Failed videos")
    
    # Individual results
    pipeline_results: List[ProcessingPipeline] = Field(default_factory=list, description="Individual pipeline results")
    
    # Batch metrics
    total_batch_time: float = Field(..., description="Total batch processing time")
    avg_processing_time: float = Field(..., description="Average processing time per video")
    throughput: float = Field(..., description="Videos processed per hour")
    
    # Quality metrics
    avg_quality_score: float = Field(..., description="Average quality score")
    quality_distribution: Dict[str, int] = Field(default_factory=dict, description="Quality distribution")
    
    # Resource usage
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage statistics")
    
    status: str = Field(..., description="Batch processing status")

class PreprocessingConfig(BaseModel):
    """Configuration cho preprocessing pipeline"""
    # Stage enables
    enable_video_processing: bool = Field(default=True, description="Enable video processing")
    enable_feature_extraction: bool = Field(default=True, description="Enable feature extraction")
    enable_knowledge_graph: bool = Field(default=True, description="Enable knowledge graph building")
    enable_vector_indexing: bool = Field(default=True, description="Enable vector indexing")
    
    # Processing strategy
    processing_mode: str = Field(default="sequential", description="Processing mode: sequential, parallel")
    parallel_workers: int = Field(default=4, description="Number of parallel workers")
    timeout_per_stage: int = Field(default=300, description="Timeout per stage (seconds)")
    max_retries: int = Field(default=3, description="Max retries per stage")
    
    # Quality thresholds
    min_video_quality: float = Field(default=0.7, description="Minimum video processing quality")
    min_feature_quality: float = Field(default=0.6, description="Minimum feature extraction quality")
    min_graph_quality: float = Field(default=0.5, description="Minimum knowledge graph quality")
    min_index_quality: float = Field(default=0.8, description="Minimum indexing quality")
    
    # Storage settings
    output_base_path: str = Field(default="./output", description="Base output path")
    keep_intermediate_files: bool = Field(default=False, description="Keep intermediate files")
    compress_outputs: bool = Field(default=True, description="Compress output files")
    
    # Performance optimization
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    batch_size: int = Field(default=1, description="Processing batch size")
    memory_limit_gb: int = Field(default=32, description="Memory limit in GB")

class PreprocessingOrchestrator(PreprocessingAgent):
    """
    Orchestrator điều phối preprocessing pipeline trong Phase 1
    Coordinate tất cả preprocessing agents để transform raw video → search-ready indexes
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="PreprocessingOrchestrator",
            **kwargs
        )
        
        # Initialize preprocessing agents
        self.video_processor = VideoProcessorAgent(**kwargs)
        self.feature_extractor = FeatureExtractorAgent(**kwargs)
        self.knowledge_graph_builder = KnowledgeGraphAgent(**kwargs)
        self.vector_indexer = VectorIndexerAgent(**kwargs)
        
        self.set_instructions([
            "You are the preprocessing pipeline orchestration coordinator.",
            "Your role is to coordinate video preprocessing from raw input to search-ready indexes:",
            "- Orchestrate video processing, feature extraction, knowledge graph building, and indexing",
            "- Ensure high-quality output at each stage",
            "- Optimize processing performance and resource usage",
            "- Handle errors gracefully and provide recovery mechanisms",
            "- Maintain comprehensive processing logs and metrics",
            "Focus on reliability, quality, and efficiency for production deployment."
        ])
        
        self.agent.response_model = ProcessingPipeline
        
    def process_video(self, 
                     video_path: str,
                     config: Optional[PreprocessingConfig] = None,
                     **kwargs) -> AgentResponse:
        """
        Process single video through complete preprocessing pipeline
        
        Args:
            video_path: Path to input video file
            config: Preprocessing configuration
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Use default config if not provided
            if config is None:
                config = PreprocessingConfig()
            
            # Initialize pipeline
            pipeline = ProcessingPipeline(
                pipeline_id=f"pipeline_{int(time.time())}",
                video_path=video_path,
                video_id=self._extract_video_id(video_path),
                current_stage=ProcessingStage.INITIALIZING,
                stage_executions=[],
                total_processing_time=0,
                success_rate=0,
                overall_quality_score=0,
                output_files=[],
                index_locations=[],
                performance_metrics={},
                status="processing"
            )
            
            # Execute pipeline stages
            pipeline = self._execute_pipeline_stages(pipeline, config, **kwargs)
            
            # Calculate final metrics
            pipeline.total_processing_time = time.time() - start_time
            pipeline.success_rate = self._calculate_success_rate(pipeline)
            pipeline.overall_quality_score = self._calculate_quality_score(pipeline)
            pipeline.performance_metrics = self._calculate_performance_metrics(pipeline)
            
            # Determine final status
            if pipeline.indexing_result and pipeline.success_rate >= 0.75:
                pipeline.current_stage = ProcessingStage.COMPLETED
                pipeline.status = "success"
            else:
                pipeline.current_stage = ProcessingStage.ERROR
                pipeline.status = "partial_success"
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="video_preprocessing",
                status="success" if pipeline.status == "success" else "partial",
                result=pipeline.dict(),
                execution_time=execution_time,
                metadata={
                    "video_path": video_path,
                    "stages_completed": len(pipeline.stage_executions),
                    "success_rate": pipeline.success_rate,
                    "quality_score": pipeline.overall_quality_score,
                    "total_time": pipeline.total_processing_time,
                    "search_ready": pipeline.indexing_result is not None
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Video preprocessing failed: {str(e)}")
            
            return self._create_response(
                task_type="video_preprocessing",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _execute_pipeline_stages(self, pipeline: ProcessingPipeline, config: PreprocessingConfig, **kwargs) -> ProcessingPipeline:
        """Execute all pipeline stages sequentially"""
        
        # Stage 1: Video Processing
        if config.enable_video_processing:
            pipeline.current_stage = ProcessingStage.VIDEO_PROCESSING
            stage_result = self._execute_stage(
                pipeline,
                "video_processing",
                self.video_processor,
                "process",
                {"video_path": pipeline.video_path},
                config.timeout_per_stage,
                **kwargs
            )
            
            if stage_result.status == "success":
                pipeline.video_processing_result = stage_result.result
                
                # Validate quality
                quality_score = self._assess_video_quality(stage_result.result)
                stage_result.output_quality = quality_score
                
                if quality_score < config.min_video_quality:
                    self.logger.warning(f"Video processing quality below threshold: {quality_score}")
        
        # Stage 2: Feature Extraction
        if config.enable_feature_extraction and pipeline.video_processing_result:
            pipeline.current_stage = ProcessingStage.FEATURE_EXTRACTION
            
            # Configure feature extraction
            feature_config = FeatureExtractionConfig(
                extract_visual=True,
                extract_audio=True,
                use_clip=True,
                use_blip=True,
                extract_ocr=True,
                use_whisper=True
            )
            
            stage_result = self._execute_stage(
                pipeline,
                "feature_extraction",
                self.feature_extractor,
                "process",
                {"video_path": pipeline.video_path, "config": feature_config},
                config.timeout_per_stage,
                **kwargs
            )
            
            if stage_result.status == "success":
                pipeline.feature_extraction_result = stage_result.result
                
                # Validate feature quality
                quality_score = self._assess_feature_quality(stage_result.result)
                stage_result.output_quality = quality_score
                
                if quality_score < config.min_feature_quality:
                    self.logger.warning(f"Feature extraction quality below threshold: {quality_score}")
        
        # Stage 3: Knowledge Graph Building
        if config.enable_knowledge_graph and pipeline.feature_extraction_result:
            pipeline.current_stage = ProcessingStage.KNOWLEDGE_GRAPH_BUILDING
            
            # Configure graph building
            graph_config = GraphBuildingConfig(
                extract_persons=True,
                extract_objects=True,
                extract_locations=True,
                extract_concepts=True,
                extract_actions=True,
                merge_similar_entities=True
            )
            
            stage_result = self._execute_stage(
                pipeline,
                "knowledge_graph_building",
                self.knowledge_graph_builder,
                "process",
                {
                    "features_data": pipeline.feature_extraction_result,
                    "config": graph_config
                },
                config.timeout_per_stage,
                **kwargs
            )
            
            if stage_result.status == "success":
                pipeline.knowledge_graph_result = stage_result.result
                
                # Validate graph quality
                quality_score = self._assess_graph_quality(stage_result.result)
                stage_result.output_quality = quality_score
                
                if quality_score < config.min_graph_quality:
                    self.logger.warning(f"Knowledge graph quality below threshold: {quality_score}")
        
        # Stage 4: Vector Indexing
        if config.enable_vector_indexing and pipeline.knowledge_graph_result and pipeline.feature_extraction_result:
            pipeline.current_stage = ProcessingStage.VECTOR_INDEXING
            
            # Configure indexing
            index_config = IndexingConfig(
                create_text_index=True,
                create_visual_index=True,
                create_audio_index=True,
                create_hybrid_index=True,
                enable_hybrid_search=True
            )
            
            stage_result = self._execute_stage(
                pipeline,
                "vector_indexing",
                self.vector_indexer,
                "process",
                {
                    "graph_data": pipeline.knowledge_graph_result,
                    "features_data": pipeline.feature_extraction_result,
                    "config": index_config
                },
                config.timeout_per_stage,
                **kwargs
            )
            
            if stage_result.status == "success":
                pipeline.indexing_result = stage_result.result
                
                # Validate index quality
                quality_score = self._assess_index_quality(stage_result.result)
                stage_result.output_quality = quality_score
                
                if quality_score < config.min_index_quality:
                    self.logger.warning(f"Vector indexing quality below threshold: {quality_score}")
        
        return pipeline
    
    def _execute_stage(self, 
                      pipeline: ProcessingPipeline,
                      stage_name: str,
                      agent: PreprocessingAgent,
                      method_name: str,
                      args: Dict[str, Any],
                      timeout: int,
                      **kwargs) -> StageExecution:
        """Execute individual pipeline stage"""
        start_time = time.time()
        
        try:
            method = getattr(agent, method_name)
            result = method(**args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            stage_execution = StageExecution(
                stage_name=stage_name,
                agent_name=agent.name,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                status=result.status,
                result=result.result,
                error_message=result.error_message,
                output_quality=None,  # Will be set by caller
                data_completeness=self._assess_data_completeness(result.result)
            )
            
            pipeline.stage_executions.append(stage_execution)
            
            self.logger.info(f"Stage {stage_name} completed in {execution_time:.2f}s with status: {result.status}")
            
            return stage_execution
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            stage_execution = StageExecution(
                stage_name=stage_name,
                agent_name=agent.name,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                status="error",
                result={},
                error_message=str(e),
                output_quality=0.0,
                data_completeness=0.0
            )
            
            pipeline.stage_executions.append(stage_execution)
            
            self.logger.error(f"Stage {stage_name} failed after {execution_time:.2f}s: {str(e)}")
            
            return stage_execution
    
    def _extract_video_id(self, video_path: str) -> str:
        """Extract video ID from path"""
        return Path(video_path).stem
    
    def _assess_video_quality(self, result: Dict[str, Any]) -> float:
        """Assess video processing quality"""
        if not result:
            return 0.0
        
        scenes = result.get("scenes", [])
        keyframes = result.get("keyframes", [])
        
        # Quality based on detection counts
        scene_score = min(len(scenes) / 10, 1.0)  # Expect ~10 scenes
        keyframe_score = min(len(keyframes) / 20, 1.0)  # Expect ~20 keyframes
        
        return (scene_score + keyframe_score) / 2
    
    def _assess_feature_quality(self, result: Dict[str, Any]) -> float:
        """Assess feature extraction quality"""
        if not result:
            return 0.0
        
        visual_features = result.get("visual_features", [])
        audio_features = result.get("audio_features", [])
        
        # Quality based on feature completeness
        visual_score = min(len(visual_features) / 30, 1.0)  # Expect ~30 visual features
        audio_score = min(len(audio_features) / 20, 1.0)  # Expect ~20 audio features
        
        # Check feature richness
        has_ocr = any(vf.get("ocr_text") for vf in visual_features)
        has_transcript = any(af.get("transcript") for af in audio_features)
        
        richness_score = (0.5 + 0.25 * has_ocr + 0.25 * has_transcript)
        
        return (visual_score + audio_score + richness_score) / 3
    
    def _assess_graph_quality(self, result: Dict[str, Any]) -> float:
        """Assess knowledge graph quality"""
        if not result:
            return 0.0
        
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        
        # Quality based on graph richness
        entity_score = min(len(entities) / 50, 1.0)  # Expect ~50 entities
        relation_score = min(len(relations) / 100, 1.0)  # Expect ~100 relations
        
        # Graph connectivity
        if entities:
            connected_entities = set()
            for relation in relations:
                connected_entities.add(relation.get("source_entity_id"))
                connected_entities.add(relation.get("target_entity_id"))
            
            connectivity_score = len(connected_entities) / len(entities)
        else:
            connectivity_score = 0.0
        
        return (entity_score + relation_score + connectivity_score) / 3
    
    def _assess_index_quality(self, result: Dict[str, Any]) -> float:
        """Assess vector indexing quality"""
        if not result:
            return 0.0
        
        text_indexes = result.get("text_indexes", [])
        visual_indexes = result.get("visual_indexes", [])
        audio_indexes = result.get("audio_indexes", [])
        hybrid_indexes = result.get("hybrid_indexes", [])
        
        # Quality based on index completeness
        text_score = min(len(text_indexes) / 20, 1.0)
        visual_score = min(len(visual_indexes) / 30, 1.0)
        audio_score = min(len(audio_indexes) / 15, 1.0)
        hybrid_score = min(len(hybrid_indexes) / 10, 1.0)
        
        return (text_score + visual_score + audio_score + hybrid_score) / 4
    
    def _assess_data_completeness(self, result: Dict[str, Any]) -> float:
        """Assess data completeness"""
        if not result:
            return 0.0
        
        # Check for required fields
        required_fields = ["status", "processing_time"]
        present_fields = sum(1 for field in required_fields if field in result)
        
        return present_fields / len(required_fields)
    
    def _calculate_success_rate(self, pipeline: ProcessingPipeline) -> float:
        """Calculate pipeline success rate"""
        if not pipeline.stage_executions:
            return 0.0
        
        successful = len([s for s in pipeline.stage_executions if s.status == "success"])
        return successful / len(pipeline.stage_executions)
    
    def _calculate_quality_score(self, pipeline: ProcessingPipeline) -> float:
        """Calculate overall quality score"""
        quality_scores = [s.output_quality for s in pipeline.stage_executions if s.output_quality is not None]
        
        if not quality_scores:
            return 0.0
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_performance_metrics(self, pipeline: ProcessingPipeline) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "total_stages": len(pipeline.stage_executions),
            "successful_stages": len([s for s in pipeline.stage_executions if s.status == "success"]),
            "avg_stage_time": sum(s.execution_time for s in pipeline.stage_executions) / max(len(pipeline.stage_executions), 1),
            "slowest_stage": max((s.execution_time for s in pipeline.stage_executions), default=0),
            "fastest_stage": min((s.execution_time for s in pipeline.stage_executions), default=0),
            "video_processing_time": next((s.execution_time for s in pipeline.stage_executions if s.stage_name == "video_processing"), 0),
            "feature_extraction_time": next((s.execution_time for s in pipeline.stage_executions if s.stage_name == "feature_extraction"), 0),
            "indexing_time": next((s.execution_time for s in pipeline.stage_executions if s.stage_name == "vector_indexing"), 0),
            "search_ready": pipeline.indexing_result is not None
        }
    
    def process_video_batch(self, 
                           video_paths: List[str],
                           config: Optional[PreprocessingConfig] = None,
                           **kwargs) -> AgentResponse:
        """Process batch of videos"""
        start_time = time.time()
        
        try:
            if config is None:
                config = PreprocessingConfig()
            
            batch_result = BatchProcessingResult(
                batch_id=f"batch_{int(time.time())}",
                total_videos=len(video_paths),
                processed_videos=0,
                failed_videos=0,
                pipeline_results=[],
                total_batch_time=0,
                avg_processing_time=0,
                throughput=0,
                avg_quality_score=0,
                quality_distribution={},
                resource_usage={},
                status="processing"
            )
            
            # Process each video
            for video_path in video_paths:
                self.logger.info(f"Processing video: {video_path}")
                
                result = self.process_video(video_path, config, **kwargs)
                
                if result.status in ["success", "partial"]:
                    batch_result.processed_videos += 1
                    pipeline_data = result.result
                    batch_result.pipeline_results.append(ProcessingPipeline(**pipeline_data))
                else:
                    batch_result.failed_videos += 1
                    self.logger.error(f"Failed to process {video_path}: {result.error_message}")
            
            # Calculate batch metrics
            batch_result.total_batch_time = time.time() - start_time
            
            if batch_result.processed_videos > 0:
                batch_result.avg_processing_time = sum(
                    p.total_processing_time for p in batch_result.pipeline_results
                ) / batch_result.processed_videos
                
                batch_result.avg_quality_score = sum(
                    p.overall_quality_score for p in batch_result.pipeline_results
                ) / batch_result.processed_videos
                
                batch_result.throughput = batch_result.processed_videos / (batch_result.total_batch_time / 3600)
            
            # Determine final status
            if batch_result.failed_videos == 0:
                batch_result.status = "success"
            elif batch_result.processed_videos > 0:
                batch_result.status = "partial_success"
            else:
                batch_result.status = "failed"
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="batch_preprocessing",
                status=batch_result.status,
                result=batch_result.dict(),
                execution_time=execution_time,
                metadata={
                    "total_videos": batch_result.total_videos,
                    "processed_videos": batch_result.processed_videos,
                    "success_rate": batch_result.processed_videos / batch_result.total_videos,
                    "avg_quality": batch_result.avg_quality_score,
                    "throughput": batch_result.throughput
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Batch preprocessing failed: {str(e)}")
            
            return self._create_response(
                task_type="batch_preprocessing",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def validate_pipeline_output(self, pipeline_result: ProcessingPipeline, **kwargs) -> AgentResponse:
        """Validate pipeline output quality"""
        
        validation_prompt = f"""
        Validate preprocessing pipeline output for video: {pipeline_result.video_id}
        
        PIPELINE EXECUTION:
        - Stages completed: {len(pipeline_result.stage_executions)}
        - Success rate: {pipeline_result.success_rate}
        - Overall quality: {pipeline_result.overall_quality_score}
        - Total time: {pipeline_result.total_processing_time}s
        
        STAGE RESULTS:
        """
        
        for stage in pipeline_result.stage_executions:
            validation_prompt += f"""
        - {stage.stage_name}: {stage.status} (quality: {stage.output_quality}, time: {stage.execution_time:.1f}s)
        """
        
        validation_prompt += """
        
        VALIDATION TASKS:
        1. Assess overall pipeline quality and completeness
        2. Identify any missing or low-quality outputs
        3. Validate search-readiness of generated indexes
        4. Check for potential issues or improvements
        5. Provide quality assurance recommendations
        
        Return comprehensive validation assessment.
        """
        
        return self.run_with_timing(validation_prompt, **kwargs)
    
    def get_preprocessing_stats(self, pipeline_results: List[ProcessingPipeline]) -> Dict[str, Any]:
        """Analyze preprocessing performance statistics"""
        
        if not pipeline_results:
            return {"error": "No pipeline results to analyze"}
        
        total_pipelines = len(pipeline_results)
        successful_pipelines = len([p for p in pipeline_results if p.status == "success"])
        
        # Performance metrics
        processing_times = [p.total_processing_time for p in pipeline_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        quality_scores = [p.overall_quality_score for p in pipeline_results]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        
        success_rates = [p.success_rate for p in pipeline_results]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        return {
            "total_pipelines": total_pipelines,
            "successful_pipelines": successful_pipelines,
            "pipeline_success_rate": successful_pipelines / total_pipelines,
            "avg_processing_time": round(avg_processing_time, 2),
            "avg_quality_score": round(avg_quality_score, 3),
            "avg_stage_success_rate": round(avg_success_rate, 3),
            "performance_distribution": {
                "fast": len([t for t in processing_times if t < 300]),  # < 5 minutes
                "medium": len([t for t in processing_times if 300 <= t < 900]),  # 5-15 minutes  
                "slow": len([t for t in processing_times if t >= 900])  # > 15 minutes
            },
            "quality_distribution": {
                "excellent": len([q for q in quality_scores if q >= 0.8]),
                "good": len([q for q in quality_scores if 0.6 <= q < 0.8]),
                "fair": len([q for q in quality_scores if 0.4 <= q < 0.6]),
                "poor": len([q for q in quality_scores if q < 0.4])
            },
            "stage_analysis": {
                "video_processing": {
                    "avg_time": sum(s.execution_time for p in pipeline_results 
                                  for s in p.stage_executions if s.stage_name == "video_processing") / total_pipelines,
                    "success_rate": len([s for p in pipeline_results for s in p.stage_executions 
                                       if s.stage_name == "video_processing" and s.status == "success"]) / total_pipelines
                },
                "feature_extraction": {
                    "avg_time": sum(s.execution_time for p in pipeline_results 
                                  for s in p.stage_executions if s.stage_name == "feature_extraction") / total_pipelines,
                    "success_rate": len([s for p in pipeline_results for s in p.stage_executions 
                                       if s.stage_name == "feature_extraction" and s.status == "success"]) / total_pipelines
                },
                "indexing": {
                    "avg_time": sum(s.execution_time for p in pipeline_results 
                                  for s in p.stage_executions if s.stage_name == "vector_indexing") / total_pipelines,
                    "success_rate": len([s for p in pipeline_results for s in p.stage_executions 
                                       if s.stage_name == "vector_indexing" and s.status == "success"]) / total_pipelines
                }
            }
        }
    
    def estimate_processing_time(self, video_paths: List[str], config: Optional[PreprocessingConfig] = None) -> Dict[str, Any]:
        """Estimate processing time cho video batch"""
        
        if config is None:
            config = PreprocessingConfig()
        
        # Basic estimation based on video file sizes và processing complexity
        total_size_gb = 0
        for video_path in video_paths:
            if Path(video_path).exists():
                size_bytes = Path(video_path).stat().st_size
                total_size_gb += size_bytes / (1024**3)
        
        # Rough estimation: 2 minutes per GB on average hardware
        base_time_per_gb = 120  # seconds
        estimated_total_time = total_size_gb * base_time_per_gb
        
        # Adjust for parallel processing
        if config.parallel_workers > 1:
            estimated_total_time = estimated_total_time / config.parallel_workers
        
        return {
            "total_videos": len(video_paths),
            "total_size_gb": round(total_size_gb, 2),
            "estimated_total_time_seconds": round(estimated_total_time),
            "estimated_total_time_hours": round(estimated_total_time / 3600, 2),
            "estimated_time_per_video": round(estimated_total_time / len(video_paths)),
            "parallel_workers": config.parallel_workers,
            "estimated_completion": "Based on average hardware performance",
            "factors_affecting_time": [
                "Video resolution and length",
                "Hardware specs (CPU/GPU)",
                "Enabled processing stages",
                "Network I/O for storage",
                "Concurrent system load"
            ]
        }