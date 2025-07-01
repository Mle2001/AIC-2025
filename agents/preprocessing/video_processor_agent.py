"""
Video Processor Agent - Xử lý video để extract scenes, keyframes, và segments
Phần của Phase 1: Preprocessing
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import time

from ..base_agent import PreprocessingAgent, AgentResponse

# Structured output models
class SceneDetection(BaseModel):
    """Kết quả scene detection"""
    scene_id: int = Field(..., description="ID của scene")
    start_time: float = Field(..., description="Thời gian bắt đầu (seconds)")
    end_time: float = Field(..., description="Thời gian kết thúc (seconds)")
    duration: float = Field(..., description="Độ dài scene (seconds)")
    confidence: float = Field(..., description="Độ tin cậy (0-1)")
    description: str = Field(..., description="Mô tả ngắn gọn về scene")

class KeyFrame(BaseModel):
    """Thông tin keyframe"""
    frame_id: int = Field(..., description="ID của frame")
    timestamp: float = Field(..., description="Thời gian trong video (seconds)")
    file_path: str = Field(..., description="Đường dẫn file keyframe")
    scene_id: int = Field(..., description="Scene chứa keyframe này")
    importance_score: float = Field(..., description="Điểm quan trọng (0-1)")
    features: Dict[str, Any] = Field(default_factory=dict, description="Visual features")

class ShotBoundary(BaseModel):
    """Shot boundary detection"""
    shot_id: int = Field(..., description="ID của shot")
    start_frame: int = Field(..., description="Frame bắt đầu")
    end_frame: int = Field(..., description="Frame kết thúc")
    start_time: float = Field(..., description="Thời gian bắt đầu")
    end_time: float = Field(..., description="Thời gian kết thúc")
    transition_type: str = Field(..., description="Loại transition: cut, fade, dissolve, etc.")
    confidence: float = Field(..., description="Độ tin cậy")

class VideoProcessingResult(BaseModel):
    """Kết quả tổng thể video processing"""
    video_id: str = Field(..., description="ID của video")
    video_path: str = Field(..., description="Đường dẫn video gốc")
    duration: float = Field(..., description="Độ dài video (seconds)")
    fps: float = Field(..., description="Frames per second")
    resolution: str = Field(..., description="Độ phân giải (WxH)")
    total_frames: int = Field(..., description="Tổng số frames")
    
    # Kết quả processing
    scenes: List[SceneDetection] = Field(default_factory=list, description="Danh sách scenes")
    keyframes: List[KeyFrame] = Field(default_factory=list, description="Danh sách keyframes")
    shot_boundaries: List[ShotBoundary] = Field(default_factory=list, description="Shot boundaries")
    
    # Metadata
    processing_time: float = Field(..., description="Thời gian xử lý")
    status: str = Field(..., description="Trạng thái xử lý")
    error_message: Optional[str] = Field(None, description="Lỗi nếu có")

class VideoProcessorAgent(PreprocessingAgent):
    """
    Agent chuyên xử lý video: scene detection, keyframe extraction, shot boundary detection
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="VideoProcessor",
            **kwargs
        )
        
        # Custom instructions cho video processing
        self.set_instructions([
            "You are a video processing specialist.",
            "Your job is to analyze videos and extract structural information.",
            "Focus on scene detection, keyframe extraction, and shot boundary detection.",
            "Always provide detailed analysis with confidence scores.",
            "Use professional video processing terminology.",
            "Be precise with timestamps and frame numbers."
        ])
        
        # Response model cho structured output
        self.agent.response_model = VideoProcessingResult
        
    def process(self, video_path: str, **kwargs) -> AgentResponse:
        """
        Xử lý video để extract scenes, keyframes, và shots
        
        Args:
            video_path: Đường dẫn tới video file
            **kwargs: Optional parameters
                - extract_keyframes: bool = True
                - detect_scenes: bool = True  
                - detect_shots: bool = True
                - keyframe_interval: int = 30 (seconds)
                - scene_threshold: float = 0.3
        """
        start_time = time.time()
        
        try:
            # Validate video path
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract video metadata
            video_info = self._get_video_info(video_path)
            
            # Prepare processing prompt
            prompt = self._create_processing_prompt(video_path, video_info, **kwargs)
            
            # Run agent với structured output
            response = self.agent.run(prompt)
            
            # Validate response is VideoProcessingResult
            if not isinstance(response, VideoProcessingResult):
                # If structured output failed, parse manually
                result = self._parse_processing_response(response.content, video_info)
            else:
                result = response
            
            execution_time = time.time() - start_time
            
            return self._create_response(
                task_type="video_processing",
                status="success",
                result=result.dict(),
                execution_time=execution_time,
                metadata={
                    "video_path": video_path,
                    "scenes_detected": len(result.scenes),
                    "keyframes_extracted": len(result.keyframes),
                    "shots_detected": len(result.shot_boundaries)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Video processing failed: {str(e)}")
            
            return self._create_response(
                task_type="video_processing", 
                status="error",
                result={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video information"""
        # Placeholder - trong thực tế sẽ dùng cv2 hoặc ffmpeg
        return {
            "duration": 120.0,  # seconds
            "fps": 30.0,
            "resolution": "1920x1080", 
            "total_frames": 3600,
            "codec": "h264",
            "bitrate": "5000kbps"
        }
    
    def _create_processing_prompt(self, video_path: str, video_info: Dict, **kwargs) -> str:
        """Tạo prompt cho video processing"""
        
        # Extract parameters
        extract_keyframes = kwargs.get('extract_keyframes', True)
        detect_scenes = kwargs.get('detect_scenes', True)
        detect_shots = kwargs.get('detect_shots', True)
        keyframe_interval = kwargs.get('keyframe_interval', 30)
        scene_threshold = kwargs.get('scene_threshold', 0.3)
        
        prompt = f"""
        Analyze the video file: {video_path}
        
        Video Information:
        - Duration: {video_info['duration']} seconds
        - FPS: {video_info['fps']}
        - Resolution: {video_info['resolution']}
        - Total Frames: {video_info['total_frames']}
        
        Processing Tasks:
        """
        
        if detect_scenes:
            prompt += f"""
        1. SCENE DETECTION:
           - Detect scene changes using threshold {scene_threshold}
           - Identify major content transitions
           - Provide confidence scores for each scene boundary
           - Generate descriptive labels for each scene
        """
        
        if extract_keyframes:
            prompt += f"""
        2. KEYFRAME EXTRACTION:
           - Extract representative keyframes every {keyframe_interval} seconds
           - Select most informative frames from each scene
           - Calculate importance scores based on visual content
           - Save keyframe metadata with timestamps
        """
        
        if detect_shots:
            prompt += f"""
        3. SHOT BOUNDARY DETECTION:
           - Detect camera cuts, fades, dissolves
           - Identify transition types between shots
           - Provide precise frame numbers for boundaries
           - Calculate confidence scores for each detection
        """
        
        prompt += """
        
        Output Requirements:
        - Return a complete VideoProcessingResult object
        - Include all detected scenes, keyframes, and shot boundaries
        - Ensure all timestamps are accurate
        - Provide confidence scores for quality assessment
        - Include any processing errors or warnings
        """
        
        return prompt
    
    def _parse_processing_response(self, response_content: str, video_info: Dict) -> VideoProcessingResult:
        """Parse response nếu structured output không work"""
        # Fallback parsing logic
        return VideoProcessingResult(
            video_id=f"video_{int(time.time())}",
            video_path="unknown",
            duration=video_info.get('duration', 0),
            fps=video_info.get('fps', 30),
            resolution=video_info.get('resolution', "unknown"),
            total_frames=video_info.get('total_frames', 0),
            scenes=[],
            keyframes=[],
            shot_boundaries=[],
            processing_time=0,
            status="parsed_fallback"
        )
    
    def process_batch_videos(self, video_paths: List[str], **kwargs) -> List[AgentResponse]:
        """Process multiple videos"""
        return self.process_batch(video_paths, **kwargs)
    
    def extract_keyframes_only(self, video_path: str, interval: int = 30) -> AgentResponse:
        """Chỉ extract keyframes không cần scene detection"""
        return self.process(
            video_path,
            extract_keyframes=True,
            detect_scenes=False,
            detect_shots=False,
            keyframe_interval=interval
        )
    
    def detect_scenes_only(self, video_path: str, threshold: float = 0.3) -> AgentResponse:
        """Chỉ detect scenes"""
        return self.process(
            video_path,
            extract_keyframes=False,
            detect_scenes=True,
            detect_shots=False,
            scene_threshold=threshold
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Lấy thống kê processing"""
        return {
            "agent_name": self.name,
            "total_sessions": len(self.agent.get_messages_for_session()) if self.agent else 0,
            "capabilities": [
                "scene_detection",
                "keyframe_extraction", 
                "shot_boundary_detection",
                "transition_detection"
            ],
            "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
            "max_duration": "unlimited",
            "batch_processing": True
        }