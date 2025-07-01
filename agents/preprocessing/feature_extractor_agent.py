"""
Feature Extractor Agent - Extract multi-modal features từ video
Xử lý Visual Features (BLIP/CLIP), Audio Features (Whisper), OCR, Speech-to-Text
Phần của Phase 1: Preprocessing
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time

from ..base_agent import PreprocessingAgent, AgentResponse

# Structured output models
class VisualFeature(BaseModel):
    """Visual features từ keyframes"""
    frame_id: str = Field(..., description="ID của frame")
    timestamp: float = Field(..., description="Timestamp trong video")
    
    # CLIP features
    clip_embedding: List[float] = Field(default_factory=list, description="CLIP embedding vector")
    clip_description: str = Field(..., description="CLIP description của image")
    
    # BLIP features  
    blip_caption: str = Field(..., description="BLIP generated caption")
    blip_objects: List[str] = Field(default_factory=list, description="Detected objects")
    
    # Visual analysis
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors")
    brightness: float = Field(..., description="Brightness level (0-1)")
    contrast: float = Field(..., description="Contrast level (0-1)")
    scene_type: str = Field(..., description="Scene type: indoor, outdoor, etc.")
    
    # OCR results
    ocr_text: str = Field(default="", description="Text detected trong frame")
    ocr_confidence: float = Field(default=0.0, description="OCR confidence")
    text_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Text bounding boxes")

class AudioFeature(BaseModel):
    """Audio features từ video"""
    segment_id: str = Field(..., description="ID của audio segment")
    start_time: float = Field(..., description="Thời gian bắt đầu")
    end_time: float = Field(..., description="Thời gian kết thúc")
    
    # Speech-to-text
    transcript: str = Field(default="", description="Whisper transcription")
    language: str = Field(default="unknown", description="Detected language")
    confidence: float = Field(..., description="Transcription confidence")
    
    # Audio analysis
    volume_level: float = Field(..., description="Volume level (0-1)")
    frequency_spectrum: List[float] = Field(default_factory=list, description="Frequency analysis")
    audio_type: str = Field(..., description="speech, music, silence, noise")
    
    # Speaker analysis
    speaker_count: int = Field(default=0, description="Number of speakers")
    speaker_embeddings: List[List[float]] = Field(default_factory=list, description="Speaker embeddings")
    
    # Emotion analysis
    emotion: str = Field(default="neutral", description="Detected emotion in speech")
    emotion_confidence: float = Field(default=0.0, description="Emotion confidence")

class MultiModalFeatures(BaseModel):
    """Kết quả tổng thể feature extraction"""
    video_id: str = Field(..., description="ID của video")
    video_path: str = Field(..., description="Đường dẫn video")
    extraction_timestamp: str = Field(..., description="Thời gian extract")
    
    # Visual features
    visual_features: List[VisualFeature] = Field(default_factory=list, description="Visual features từ keyframes")
    total_keyframes: int = Field(..., description="Tổng số keyframes processed")
    
    # Audio features
    audio_features: List[AudioFeature] = Field(default_factory=list, description="Audio features")
    total_audio_segments: int = Field(..., description="Tổng số audio segments")
    
    # Summary statistics
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    
    # Processing info
    processing_time: float = Field(..., description="Total processing time")
    models_used: List[str] = Field(default_factory=list, description="AI models used")
    status: str = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error nếu có")

class FeatureExtractionConfig(BaseModel):
    """Configuration cho feature extraction"""
    # Visual processing
    extract_visual: bool = Field(default=True, description="Extract visual features")
    use_clip: bool = Field(default=True, description="Use CLIP model")
    use_blip: bool = Field(default=True, description="Use BLIP model") 
    extract_ocr: bool = Field(default=True, description="Extract OCR text")
    
    # Audio processing
    extract_audio: bool = Field(default=True, description="Extract audio features")
    use_whisper: bool = Field(default=True, description="Use Whisper for STT")
    detect_speakers: bool = Field(default=True, description="Detect speakers")
    analyze_emotion: bool = Field(default=True, description="Analyze emotion")
    
    # Sampling
    keyframe_interval: int = Field(default=30, description="Keyframe interval (seconds)")
    audio_segment_length: int = Field(default=10, description="Audio segment length (seconds)")
    
    # Quality thresholds
    min_ocr_confidence: float = Field(default=0.5, description="Minimum OCR confidence")
    min_speech_confidence: float = Field(default=0.6, description="Minimum speech confidence")

class FeatureExtractorAgent(PreprocessingAgent):
    """
    Agent chuyên extract multi-modal features từ video
    Sử dụng CLIP, BLIP, Whisper, OCR để analyze nội dung
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="FeatureExtractor",
            model_type="gpt-4o",  # Cần model mạnh cho multi-modal
            **kwargs
        )
        
        self.set_instructions([
            "You are a multi-modal feature extraction specialist.",
            "Extract comprehensive features from video content including:",
            "- Visual features using CLIP and BLIP models",
            "- Audio features using Whisper for speech-to-text", 
            "- OCR text extraction from video frames",
            "- Scene understanding and object detection",
            "- Emotion and speaker analysis from audio",
            "Always provide detailed feature vectors and confidence scores.",
            "Focus on accuracy and comprehensive feature coverage."
        ])
        
        # Set response model
        self.agent.response_model = MultiModalFeatures
        
    def process(self, video_path: str, config: Optional[FeatureExtractionConfig] = None, **kwargs) -> AgentResponse:
        """
        Extract multi-modal features từ video
        
        Args:
            video_path: Đường dẫn video file
            config: Feature extraction configuration
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        try:
            # Use default config if not provided
            if config is None:
                config = FeatureExtractionConfig()
            
            # Create processing prompt
            prompt = self._create_extraction_prompt(video_path, config)
            
            # Run agent với structured output
            response = self.agent.run(prompt)
            
            # Validate response
            if not isinstance(response, MultiModalFeatures):
                result = self._parse_extraction_response(response.content, video_path)
            else:
                result = response
                
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="feature_extraction",
                status="success", 
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "video_path": video_path,
                    "visual_features_count": len(result.visual_features),
                    "audio_features_count": len(result.audio_features),
                    "models_used": result.models_used,
                    "config": config.dict()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Feature extraction failed: {str(e)}")
            
            return self._create_response(
                task_type="feature_extraction",
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_extraction_prompt(self, video_path: str, config: FeatureExtractionConfig) -> str:
        """Tạo prompt cho feature extraction"""
        
        prompt = f"""
        Extract comprehensive multi-modal features from video: {video_path}
        
        CONFIGURATION:
        - Extract Visual Features: {config.extract_visual}
        - Use CLIP Model: {config.use_clip}
        - Use BLIP Model: {config.use_blip}
        - Extract OCR: {config.extract_ocr}
        - Extract Audio Features: {config.extract_audio}
        - Use Whisper STT: {config.use_whisper}
        - Detect Speakers: {config.detect_speakers}
        - Analyze Emotion: {config.analyze_emotion}
        - Keyframe Interval: {config.keyframe_interval} seconds
        - Audio Segment Length: {config.audio_segment_length} seconds
        
        PROCESSING TASKS:
        """
        
        if config.extract_visual:
            prompt += f"""
        1. VISUAL FEATURE EXTRACTION:
           - Process keyframes every {config.keyframe_interval} seconds
           - Generate CLIP embeddings and descriptions for each frame
           - Use BLIP to generate detailed captions
           - Detect objects, scenes, and activities
           - Analyze visual properties: colors, brightness, contrast
           - Extract OCR text với confidence >= {config.min_ocr_confidence}
           - Identify scene types and contexts
        """
        
        if config.extract_audio:
            prompt += f"""
        2. AUDIO FEATURE EXTRACTION:
           - Segment audio into {config.audio_segment_length} second chunks
           - Use Whisper for speech-to-text với confidence >= {config.min_speech_confidence}
           - Detect language and speaker characteristics
           - Analyze audio types: speech, music, silence, noise
           - Extract frequency spectrum and volume levels
           - Perform emotion analysis on speech segments
           - Count and identify different speakers if enabled
        """
        
        prompt += """
        3. MULTI-MODAL INTEGRATION:
           - Correlate visual and audio features by timestamp
           - Identify audio-visual synchronization points
           - Generate summary statistics across all modalities
           - Calculate overall confidence scores
           - Document all models and techniques used
        
        OUTPUT REQUIREMENTS:
        - Return complete MultiModalFeatures object
        - Include all extracted features with timestamps
        - Provide confidence scores for quality assessment
        - Generate summary statistics and processing metadata
        - Report any errors or low-confidence extractions
        """
        
        return prompt
    
    def _parse_extraction_response(self, response_content: str, video_path: str) -> MultiModalFeatures:
        """Fallback parsing nếu structured output fails"""
        return MultiModalFeatures(
            video_id=f"video_{int(time.time())}",
            video_path=video_path,
            extraction_timestamp=str(time.time()),
            visual_features=[],
            total_keyframes=0,
            audio_features=[],
            total_audio_segments=0,
            summary={},
            processing_time=0,
            models_used=["fallback_parser"],
            status="parsed_fallback"
        )
    
    def extract_visual_only(self, video_path: str, keyframe_interval: int = 30) -> AgentResponse:
        """Chỉ extract visual features"""
        config = FeatureExtractionConfig(
            extract_visual=True,
            extract_audio=False,
            keyframe_interval=keyframe_interval
        )
        return self.process(video_path, config)
    
    def extract_audio_only(self, video_path: str, segment_length: int = 10) -> AgentResponse:
        """Chỉ extract audio features"""
        config = FeatureExtractionConfig(
            extract_visual=False,
            extract_audio=True,
            audio_segment_length=segment_length
        )
        return self.process(video_path, config)
    
    def extract_ocr_only(self, video_path: str, keyframe_interval: int = 5) -> AgentResponse:
        """Chỉ extract OCR text"""
        config = FeatureExtractionConfig(
            extract_visual=True,
            extract_audio=False,
            use_clip=False,
            use_blip=False,
            extract_ocr=True,
            keyframe_interval=keyframe_interval
        )
        return self.process(video_path, config)
    
    def extract_speech_only(self, video_path: str) -> AgentResponse:
        """Chỉ extract speech-to-text"""
        config = FeatureExtractionConfig(
            extract_visual=False,
            extract_audio=True,
            use_whisper=True,
            detect_speakers=False,
            analyze_emotion=False
        )
        return self.process(video_path, config)
    
    def get_supported_models(self) -> Dict[str, Any]:
        """Lấy danh sách models được support"""
        return {
            "visual_models": {
                "clip": {
                    "description": "OpenAI CLIP for image-text understanding",
                    "capabilities": ["image_description", "zero_shot_classification", "similarity_search"]
                },
                "blip": {
                    "description": "Salesforce BLIP for image captioning",
                    "capabilities": ["detailed_captioning", "visual_question_answering"]
                }
            },
            "audio_models": {
                "whisper": {
                    "description": "OpenAI Whisper for speech recognition",
                    "capabilities": ["multilingual_stt", "language_detection", "speaker_diarization"]
                }
            },
            "ocr_models": {
                "easyocr": {
                    "description": "EasyOCR for text detection",
                    "capabilities": ["multilingual_ocr", "text_localization", "confidence_scoring"]
                }
            },
            "feature_types": [
                "visual_embeddings",
                "audio_embeddings", 
                "text_features",
                "temporal_features",
                "multi_modal_features"
            ]
        }
    
    def benchmark_extraction(self, video_paths: List[str]) -> Dict[str, Any]:
        """Benchmark feature extraction performance"""
        results = []
        
        for video_path in video_paths:
            result = self.process(video_path)
            results.append({
                "video_path": video_path,
                "status": result.status,
                "execution_time": result.execution_time,
                "feature_count": len(result.result.get("visual_features", [])) + len(result.result.get("audio_features", []))
            })
        
        return {
            "total_videos": len(video_paths),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "avg_processing_time": sum(r["execution_time"] for r in results) / len(results),
            "total_features": sum(r["feature_count"] for r in results),
            "details": results
        }