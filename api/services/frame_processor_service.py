"""
Frame Sequence Processing Service
=================================

Integrated service that combines VideoRAG frame sequence processing
with competition query functionality for the AIC-2025 system.
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np

# MoviePy for frame sequence processing
try:
    from moviepy.editor import ImageSequenceClip, VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available - frame sequence processing disabled")

# VideoRAG imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'VideoRAG'))

try:
    from videorag import VideoRAG, QueryParam
    from videorag._llm import openai_4o_mini_config
    VIDEORAG_AVAILABLE = True
except ImportError as e:
    VIDEORAG_AVAILABLE = False
    logging.warning(f"VideoRAG not available: {e}")

from api.config import get_openai_api_key, settings
from fastapi import HTTPException

warnings.filterwarnings("ignore")

class FrameSequenceClip:
    """
    Adapter that makes frame sequences compatible with VideoRAG
    """
    
    def __init__(self, frames_folder: str, fps: float = 30.0, audio_file: Optional[str] = None):
        if not MOVIEPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="MoviePy not available for frame processing")
        
        self.frames_folder = Path(frames_folder)
        self.fps = fps
        self.audio_file = audio_file
        
        # Discover frame files
        self.frame_files = self._discover_frames()
        
        # Create MoviePy ImageSequenceClip
        self.clip = ImageSequenceClip(
            sequence=str(self.frames_folder),
            fps=fps
        )
        
        # Add audio if provided
        if audio_file and Path(audio_file).exists():
            from moviepy.editor import AudioFileClip
            audio_clip = AudioFileClip(audio_file)
            self.clip = self.clip.set_audio(audio_clip)
        
        # Set attributes
        self.duration = self.clip.duration
        self.size = self.clip.size
        
        # Create temporary video file
        self.temp_video_path = self._create_temp_video()
    
    def _discover_frames(self) -> List[Path]:
        """Discover all frame files in the folder"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        frame_files = []
        
        for ext in extensions:
            frame_files.extend(self.frames_folder.glob(f'*{ext}'))
            frame_files.extend(self.frames_folder.glob(f'*{ext.upper()}'))
        
        frame_files.sort(key=lambda x: x.name)
        
        if not frame_files:
            raise ValueError(f"No frame files found in {self.frames_folder}")
        
        return frame_files
    
    def _create_temp_video(self) -> str:
        """Create temporary video file for VideoRAG processing"""
        temp_dir = Path(settings.upload_dir) / "temp_videos"
        temp_dir.mkdir(exist_ok=True)
        
        video_name = self.frames_folder.name
        temp_video_path = temp_dir / f"{video_name}.mp4"
        
        if not temp_video_path.exists():
            self.clip.write_videofile(
                str(temp_video_path),
                codec='libx264',
                audio_codec='aac' if self.audio_file else None,
                verbose=False,
                logger=None
            )
        
        return str(temp_video_path)
    
    def get_metadata(self) -> Dict:
        """Get metadata about the frame sequence"""
        return {
            'frames_folder': str(self.frames_folder),
            'frame_count': len(self.frame_files),
            'fps': self.fps,
            'duration': self.duration,
            'size': self.size,
            'audio_file': self.audio_file,
            'temp_video_path': self.temp_video_path
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_video_path') and Path(self.temp_video_path).exists():
            os.remove(self.temp_video_path)


class CompetitionQueryEngine:
    """
    Optimized query engine for competition phase
    """
    
    def __init__(self, index_dir: str, fast_mode: bool = True):
        if not VIDEORAG_AVAILABLE:
            raise HTTPException(status_code=500, detail="VideoRAG not available")
        
        self.index_dir = Path(index_dir)
        self.fast_mode = fast_mode
        
        # Setup environment
        openai_key = get_openai_api_key()
        if not openai_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Setup logging
        self.logger = logging.getLogger('CompetitionEngine')
        
        # Load VideoRAG
        self._load_videorag()
        self._load_metadata()
        self._setup_query_params()
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.query_history = []
    
    def _load_videorag(self):
        """Load VideoRAG with pre-built index"""
        try:
            self.videorag = VideoRAG(
                llm=openai_4o_mini_config,
                working_dir=str(self.index_dir)
            )
            self.videorag.load_caption_model(debug=False)
            self.logger.info("VideoRAG loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load VideoRAG: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load VideoRAG: {str(e)}")
    
    def _load_metadata(self):
        """Load processing metadata"""
        manifest_path = self.index_dir / 'frame_sequences_manifest.json'
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _setup_query_params(self):
        """Setup optimized query parameters"""
        self.fast_param = QueryParam(mode="videorag")
        
        if self.fast_mode:
            self.fast_param.wo_reference = True
            self.fast_param.max_retrieval_results = 5
            self.fast_param.temperature = 0.1
        else:
            self.fast_param.wo_reference = False
            self.fast_param.max_retrieval_results = 10
            self.fast_param.temperature = 0.3
    
    def query(self, query_text: str, timeout: float = 30.0) -> Dict:
        """Execute competition query with timeout"""
        start_time = time.time()
        
        try:
            response = self.videorag.query(
                query=query_text,
                param=self.fast_param
            )
            
            query_time = time.time() - start_time
            
            # Parse response
            parsed_result = self._parse_response_timestamps(response)
            
            result = {
                'query': query_text,
                'video_folder': parsed_result['video_folder'],
                'start_timestamp': parsed_result['start_time'],
                'end_timestamp': parsed_result['end_time'],
                'confidence': parsed_result['confidence'],
                'query_time': query_time,
                'status': 'success',
                'raw_response': str(response)
            }
            
            # Update statistics
            self.query_count += 1
            self.total_query_time += query_time
            self.query_history.append(result)
            
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            
            error_result = {
                'query': query_text,
                'video_folder': None,
                'start_timestamp': None,
                'end_timestamp': None,
                'confidence': 0.0,
                'query_time': query_time,
                'status': 'failed',
                'error': str(e)
            }
            
            self.logger.error(f"Query failed: {e}")
            return error_result
    
    def _parse_response_timestamps(self, response: str) -> Dict:
        """Parse VideoRAG response to extract timestamps and video information"""
        import re
        
        result = {
            'video_folder': None,
            'start_time': None,
            'end_time': None,
            'confidence': 0.5,
            'raw_response': response
        }
        
        try:
            response_text = str(response).lower()
            
            # Extract video/folder references
            video_patterns = [
                r'video[_\s]*(\d+)',
                r'folder[_\s]*(\d+)',
                r'sequence[_\s]*(\d+)',
                r'clip[_\s]*(\d+)',
                r'([a-zA-Z0-9_-]+\.mp4)',
                r'([a-zA-Z0-9_-]+\.mkv)',
                r'([a-zA-Z0-9_-]+)/',
            ]
            
            for pattern in video_patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    result['video_folder'] = matches[0]
                    break
            
            # Extract timestamps
            time_patterns = [
                r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
                r'(\d{1,2}):(\d{2})',           # MM:SS
                r'(\d+\.?\d*)\s*seconds?',       # seconds
                r'(\d+\.?\d*)\s*s\b',           # s
                r'frame[s]?\s*(\d+)',           # frames
                r'from\s*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)',  # ranges
                r'at\s*(\d+\.?\d*)\s*minutes?', # minutes
            ]
            
            times_found = []
            
            for pattern in time_patterns:
                matches = re.findall(pattern, response_text)
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) == 3:  # HH:MM:SS
                            h, m, s = map(float, match)
                            total_seconds = h * 3600 + m * 60 + s
                        elif len(match) == 2:  # MM:SS or range
                            try:
                                m, s = map(float, match)
                                total_seconds = m * 60 + s
                            except:
                                total_seconds = float(match[0])
                                times_found.append(float(match[1]))
                        else:
                            total_seconds = float(match[0])
                    else:
                        total_seconds = float(match)
                    
                    times_found.append(total_seconds)
            
            # Set start and end times
            if times_found:
                times_found.sort()
                result['start_time'] = times_found[0]
                
                if len(times_found) > 1:
                    result['end_time'] = times_found[-1]
                else:
                    result['end_time'] = result['start_time'] + 10.0
                
                result['confidence'] = 0.8
            
            # Convert frame numbers to timestamps if no time found
            if not times_found:
                frame_pattern = r'frame[s]?\s*(\d+)'
                frame_matches = re.findall(frame_pattern, response_text)
                
                if frame_matches:
                    fps = self.metadata.get('sequences', [{}])[0].get('metadata', {}).get('fps', 30.0)
                    start_frame = int(frame_matches[0])
                    result['start_time'] = start_frame / fps
                    result['end_time'] = result['start_time'] + 10.0
                    result['confidence'] = 0.7
        
        except Exception as e:
            self.logger.warning(f"Error parsing response: {e}")
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if self.query_count == 0:
            return {'message': 'No queries processed yet'}
        
        avg_time = self.total_query_time / self.query_count
        success_rate = sum(1 for q in self.query_history if q['status'] == 'success') / self.query_count
        
        return {
            'total_queries': self.query_count,
            'average_query_time': avg_time,
            'total_time': self.total_query_time,
            'success_rate': success_rate,
            'mode': 'FAST' if self.fast_mode else 'ACCURATE'
        }


class FrameProcessorService:
    """
    Integrated frame sequence processor service
    """
    
    def __init__(self):
        self.frames_root_dir = Path(settings.upload_dir) / "frame_sequences"
        self.output_dir = Path(settings.upload_dir) / "frame_index"
        self.default_fps = 30.0
        
        # Create directories
        self.frames_root_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('FrameProcessor')
        
        # VideoRAG instance
        self.videorag = None
        self.query_engine = None
        
        # Track processed sequences
        self.processed_sequences = []
        self.frame_clips = []
    
    async def upload_frame_sequence(self, folder_name: str, frame_files: List[Any]) -> Dict:
        """
        Upload frame sequence files
        
        Args:
            folder_name: Name of the video folder
            frame_files: List of uploaded frame files
            
        Returns:
            Upload result
        """
        try:
            # Create folder for this sequence
            sequence_folder = self.frames_root_dir / folder_name
            sequence_folder.mkdir(exist_ok=True)
            
            # Save frame files
            saved_files = []
            for i, frame_file in enumerate(frame_files):
                frame_filename = f"frame_{i+1:04d}.jpg"
                frame_path = sequence_folder / frame_filename
                
                async with aiofiles.open(frame_path, 'wb') as f:
                    content = await frame_file.read()
                    await f.write(content)
                
                saved_files.append(str(frame_path))
            
            return {
                'status': 'success',
                'folder_name': folder_name,
                'frame_count': len(saved_files),
                'folder_path': str(sequence_folder),
                'saved_files': saved_files
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading frame sequence: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def process_frame_sequence(self, folder_name: str, fps: Optional[float] = None) -> Dict:
        """
        Process a single frame sequence folder
        
        Args:
            folder_name: Name of the folder containing frames
            fps: FPS to use for processing
            
        Returns:
            Processing result
        """
        if not MOVIEPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="MoviePy not available")
        
        start_time = time.time()
        
        try:
            frames_folder = self.frames_root_dir / folder_name
            if not frames_folder.exists():
                raise FileNotFoundError(f"Frame sequence folder not found: {folder_name}")
            
            # Create frame sequence clip
            fps = fps or self.default_fps
            frame_clip = FrameSequenceClip(
                frames_folder=str(frames_folder),
                fps=fps
            )
            
            # Get metadata
            metadata = frame_clip.get_metadata()
            
            # Store clip for later processing
            self.frame_clips.append(frame_clip)
            
            result = {
                'status': 'success',
                'folder_name': folder_name,
                'metadata': metadata,
                'processing_time': time.time() - start_time,
                'temp_video_path': frame_clip.get_video_path()
            }
            
            self.processed_sequences.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {folder_name}: {e}")
            return {
                'status': 'failed',
                'folder_name': folder_name,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def build_videorag_index(self) -> Dict:
        """
        Build VideoRAG index from processed frame sequences
        """
        if not VIDEORAG_AVAILABLE:
            raise HTTPException(status_code=500, detail="VideoRAG not available")
        
        try:
            # Collect all temporary video paths
            video_paths = []
            for result in self.processed_sequences:
                if result['status'] == 'success':
                    video_paths.append(result['temp_video_path'])
            
            if not video_paths:
                raise ValueError("No successfully processed frame sequences found!")
            
            # Initialize VideoRAG
            openai_key = get_openai_api_key()
            if not openai_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            
            os.environ["OPENAI_API_KEY"] = openai_key
            
            self.videorag = VideoRAG(
                llm=openai_4o_mini_config,
                working_dir=str(self.output_dir)
            )
            
            # Load models
            self.videorag.load_caption_model(debug=False)
            
            # Insert videos into VideoRAG
            self.videorag.insert_video(video_path_list=video_paths)
            
            # Save processing manifest
            await self._save_processing_manifest()
            
            return {
                'status': 'success',
                'videos_processed': len(video_paths),
                'index_path': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error building VideoRAG index: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def query_frame_sequences(self, query: str, fast_mode: bool = True) -> Dict:
        """
        Query the processed frame sequences
        
        Args:
            query: Search query
            fast_mode: Use fast mode for queries
            
        Returns:
            Query result
        """
        try:
            # Initialize query engine if not exists
            if not self.query_engine:
                self.query_engine = CompetitionQueryEngine(
                    index_dir=str(self.output_dir),
                    fast_mode=fast_mode
                )
            
            # Execute query
            result = self.query_engine.query(query)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying frame sequences: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _save_processing_manifest(self):
        """Save processing manifest with metadata"""
        manifest = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'frames_root_dir': str(self.frames_root_dir),
            'total_sequences': len(self.processed_sequences),
            'successful_sequences': sum(1 for r in self.processed_sequences if r['status'] == 'success'),
            'failed_sequences': sum(1 for r in self.processed_sequences if r['status'] == 'failed'),
            'default_fps': self.default_fps,
            'sequences': self.processed_sequences
        }
        
        manifest_path = self.output_dir / 'frame_sequences_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    async def cleanup_temp_files(self):
        """Clean up temporary video files"""
        for clip in self.frame_clips:
            clip.cleanup()
        
        # Remove temp directory if empty
        temp_dir = Path(settings.upload_dir) / "temp_videos"
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    
    async def get_processing_status(self) -> Dict:
        """Get current processing status"""
        manifest_path = self.output_dir / 'frame_sequences_manifest.json'
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            return {
                'status': 'ready',
                'total_sequences': manifest['total_sequences'],
                'successful_sequences': manifest['successful_sequences'],
                'failed_sequences': manifest['failed_sequences'],
                'last_updated': manifest['timestamp']
            }
        else:
            return {
                'status': 'not_ready',
                'message': 'No frame sequences processed yet'
            }
    
    async def discover_frame_folders(self) -> List[str]:
        """Discover existing frame sequence folders"""
        folders = []
        
        for item in self.frames_root_dir.iterdir():
            if item.is_dir():
                # Check if folder contains image files
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_files.extend(item.glob(f'*{ext}'))
                    image_files.extend(item.glob(f'*{ext.upper()}'))
                
                if image_files:
                    folders.append(item.name)
        
        return folders
    
    async def get_query_stats(self) -> Dict:
        """Get query performance statistics"""
        if not self.query_engine:
            return {'message': 'Query engine not initialized'}
        
        return self.query_engine.get_performance_stats()


# Global service instance
frame_processor_service = FrameProcessorService()
