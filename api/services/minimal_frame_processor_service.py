"""
Minimal Frame Processor Service
==============================

A minimal version of the frame processor service that doesn't depend on heavy libraries
until they are actually needed.
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

from api.config import get_openai_api_key, settings
from fastapi import HTTPException

warnings.filterwarnings("ignore")

class MinimalFrameProcessorService:
    """
    Minimal frame sequence processor service that loads dependencies lazily
    """
    
    def __init__(self):
        self.frames_root_dir = Path(settings.upload_dir) / "frame_sequences"
        self.output_dir = Path(settings.upload_dir) / "frame_index"
        self.default_fps = 30.0
        
        # Create directories
        self.frames_root_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('MinimalFrameProcessor')
        
        # Lazy loading flags
        self._moviepy_available = None
        self._videorag_available = None
        self._query_engine = None
        
        # Track processed sequences
        self.processed_sequences = []
        self.frame_clips = []
    
    def _check_moviepy(self):
        """Check if MoviePy is available"""
        if self._moviepy_available is None:
            try:
                from moviepy.editor import ImageSequenceClip, VideoFileClip
                self._moviepy_available = True
            except ImportError:
                self._moviepy_available = False
                self.logger.warning("MoviePy not available - frame sequence processing disabled")
        return self._moviepy_available
    
    def _check_videorag(self):
        """Check if VideoRAG is available"""
        if self._videorag_available is None:
            try:
                # Add VideoRAG to path if needed
                videorag_path = os.path.join(os.path.dirname(__file__), '..', '..', 'VideoRAG')
                if videorag_path not in sys.path:
                    sys.path.append(videorag_path)
                
                from videorag import VideoRAG, QueryParam
                from videorag._llm import openai_4o_mini_config
                self._videorag_available = True
            except ImportError as e:
                self._videorag_available = False
                self.logger.warning(f"VideoRAG not available: {e}")
        return self._videorag_available
    
    async def upload_frame_sequence(self, folder_name: str, frame_files: List[Any]) -> Dict:
        """Upload frame sequence files"""
        try:
            # Create folder for this sequence
            sequence_folder = self.frames_root_dir / folder_name
            sequence_folder.mkdir(exist_ok=True)
            
            # Save frame files
            saved_files = []
            for i, frame_file in enumerate(frame_files):
                frame_filename = f"frame_{i+1:04d}.jpg"
                frame_path = sequence_folder / frame_filename
                
                # Handle different file types
                if hasattr(frame_file, 'read'):
                    # UploadFile object
                    content = await frame_file.read()
                    with open(frame_path, 'wb') as f:
                        f.write(content)
                else:
                    # File path or bytes
                    with open(frame_path, 'wb') as f:
                        f.write(frame_file)
                
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
        """Process a single frame sequence folder"""
        if not self._check_moviepy():
            return {
                'status': 'failed',
                'error': 'MoviePy not available - cannot process frame sequences'
            }
        
        start_time = time.time()
        
        try:
            frames_folder = self.frames_root_dir / folder_name
            if not frames_folder.exists():
                return {
                    'status': 'failed',
                    'error': f"Frame sequence folder not found: {folder_name}"
                }
            
            # Import MoviePy components
            from moviepy.editor import ImageSequenceClip
            
            # Create frame sequence clip
            fps = fps or self.default_fps
            clip = ImageSequenceClip(
                sequence=str(frames_folder),
                fps=fps
            )
            
            # Create temporary video file
            temp_dir = Path(settings.upload_dir) / "temp_videos"
            temp_dir.mkdir(exist_ok=True)
            
            temp_video_path = temp_dir / f"{folder_name}.mp4"
            
            if not temp_video_path.exists():
                clip.write_videofile(
                    str(temp_video_path),
                    codec='libx264',
                    verbose=False,
                    logger=None
                )
            
            result = {
                'status': 'success',
                'folder_name': folder_name,
                'metadata': {
                    'frames_folder': str(frames_folder),
                    'fps': fps,
                    'duration': clip.duration,
                    'temp_video_path': str(temp_video_path)
                },
                'processing_time': time.time() - start_time,
                'temp_video_path': str(temp_video_path)
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
        """Build VideoRAG index from processed frame sequences"""
        if not self._check_videorag():
            return {
                'status': 'failed',
                'error': 'VideoRAG not available - cannot build index'
            }
        
        try:
            # Import VideoRAG components
            from videorag import VideoRAG
            from videorag._llm import openai_4o_mini_config
            
            # Collect all temporary video paths
            video_paths = []
            for result in self.processed_sequences:
                if result['status'] == 'success':
                    video_paths.append(result['temp_video_path'])
            
            if not video_paths:
                return {
                    'status': 'failed',
                    'error': 'No successfully processed frame sequences found!'
                }
            
            # Setup OpenAI API key
            openai_key = get_openai_api_key()
            if not openai_key:
                return {
                    'status': 'failed',
                    'error': 'OpenAI API key not configured'
                }
            
            os.environ["OPENAI_API_KEY"] = openai_key
            
            # Initialize VideoRAG
            videorag = VideoRAG(
                llm=openai_4o_mini_config,
                working_dir=str(self.output_dir)
            )
            
            # Load models
            videorag.load_caption_model(debug=False)
            
            # Insert videos into VideoRAG
            videorag.insert_video(video_path_list=video_paths)
            
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
        """Query the processed frame sequences"""
        if not self._check_videorag():
            return {
                'status': 'failed',
                'error': 'VideoRAG not available - cannot query'
            }
        
        try:
            # Import VideoRAG components
            from videorag import VideoRAG, QueryParam
            from videorag._llm import openai_4o_mini_config
            
            # Initialize query engine if not exists
            if not self._query_engine:
                # Setup OpenAI API key
                openai_key = get_openai_api_key()
                if not openai_key:
                    return {
                        'status': 'failed',
                        'error': 'OpenAI API key not configured'
                    }
                
                os.environ["OPENAI_API_KEY"] = openai_key
                
                # Initialize VideoRAG
                videorag = VideoRAG(
                    llm=openai_4o_mini_config,
                    working_dir=str(self.output_dir)
                )
                videorag.load_caption_model(debug=False)
                
                self._query_engine = videorag
            
            # Setup query parameters
            param = QueryParam(mode="videorag")
            if fast_mode:
                param.wo_reference = True
                param.max_retrieval_results = 5
                param.temperature = 0.1
            else:
                param.wo_reference = False
                param.max_retrieval_results = 10
                param.temperature = 0.3
            
            # Execute query
            start_time = time.time()
            response = self._query_engine.query(query=query, param=param)
            query_time = time.time() - start_time
            
            # Basic response parsing
            result = {
                'status': 'success',
                'query': query,
                'video_folder': None,
                'start_timestamp': None,
                'end_timestamp': None,
                'confidence': 0.5,
                'query_time': query_time,
                'raw_response': str(response)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying frame sequences: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def get_processing_status(self) -> Dict:
        """Get current processing status"""
        manifest_path = self.output_dir / 'frame_sequences_manifest.json'
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                return {
                    'status': 'ready',
                    'total_sequences': manifest['total_sequences'],
                    'successful_sequences': manifest['successful_sequences'],
                    'failed_sequences': manifest['failed_sequences'],
                    'last_updated': manifest['timestamp']
                }
            except Exception as e:
                self.logger.error(f"Error reading manifest: {e}")
                return {
                    'status': 'error',
                    'message': f'Error reading manifest: {str(e)}'
                }
        else:
            return {
                'status': 'not_ready',
                'message': 'No frame sequences processed yet'
            }
    
    async def discover_frame_folders(self) -> List[str]:
        """Discover existing frame sequence folders"""
        folders = []
        
        try:
            for item in self.frames_root_dir.iterdir():
                if item.is_dir():
                    # Check if folder contains image files
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        image_files.extend(item.glob(f'*{ext}'))
                        image_files.extend(item.glob(f'*{ext.upper()}'))
                    
                    if image_files:
                        folders.append(item.name)
        except Exception as e:
            self.logger.error(f"Error discovering folders: {e}")
        
        return folders
    
    async def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_dir = Path(settings.upload_dir) / "temp_videos"
            if temp_dir.exists():
                for file in temp_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                
                # Remove directory if empty
                if not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")
    
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


# Global service instance
minimal_frame_processor_service = MinimalFrameProcessorService()
