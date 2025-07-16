#!/usr/bin/env python3
"""
VideoRAG Frame Sequence Adapter
==============================

Adapter to modify VideoRAG for processing frame sequences from folders
instead of traditional video files.

Competition Usage:
    competition_data/
    ├── video1/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   └── ...
    ├── video2/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   └── ...
    └── ...

Author: Competition Team
"""

import os
import time
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union
import multiprocessing
from tqdm import tqdm
import numpy as np

# MoviePy for frame sequence processing
from moviepy.editor import ImageSequenceClip, VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip as OriginalVideoFileClip

# VideoRAG imports
from videorag import VideoRAG, QueryParam
from videorag._llm import openai_4o_mini_config

warnings.filterwarnings("ignore")

class FrameSequenceClip:
    """
    Custom adapter that makes frame sequences compatible with VideoRAG
    """
    
    def __init__(self, frames_folder: str, fps: float = 30.0, audio_file: Optional[str] = None):
        """
        Initialize frame sequence as video clip
        
        Args:
            frames_folder: Folder containing frame images
            fps: Frames per second to assume for the sequence
            audio_file: Optional audio file to associate with frames
        """
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
        
        # Set attributes to mimic video file
        self.duration = self.clip.duration
        self.size = self.clip.size
        
        # Create temporary video file for VideoRAG compatibility
        self.temp_video_path = self._create_temp_video()
    
    def _discover_frames(self) -> List[Path]:
        """Discover all frame files in the folder"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        frame_files = []
        
        for ext in extensions:
            frame_files.extend(self.frames_folder.glob(f'*{ext}'))
            frame_files.extend(self.frames_folder.glob(f'*{ext.upper()}'))
        
        # Sort alphanumerically
        frame_files.sort(key=lambda x: x.name)
        
        if not frame_files:
            raise ValueError(f"No frame files found in {self.frames_folder}")
        
        return frame_files
    
    def _create_temp_video(self) -> str:
        """Create temporary video file for VideoRAG processing"""
        temp_dir = Path("./temp_videos")
        temp_dir.mkdir(exist_ok=True)
        
        # Use folder name as video name
        video_name = self.frames_folder.name
        temp_video_path = temp_dir / f"{video_name}.mp4"
        
        # Only create if doesn't exist
        if not temp_video_path.exists():
            print(f"Creating temporary video: {temp_video_path}")
            self.clip.write_videofile(
                str(temp_video_path),
                codec='libx264',
                audio_codec='aac' if self.audio_file else None,
                verbose=False,
                logger=None
            )
        
        return str(temp_video_path)
    
    def get_video_path(self) -> str:
        """Get the path to the temporary video file"""
        return self.temp_video_path
    
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


class VideoRAGFrameProcessor:
    """
    Modified VideoRAG processor for frame sequences
    """
    
    def __init__(self, 
                 frames_root_dir: str,
                 output_dir: str,
                 openai_key: str,
                 default_fps: float = 30.0):
        """
        Initialize the frame sequence processor
        
        Args:
            frames_root_dir: Root directory containing video folders with frames
            output_dir: Output directory for VideoRAG indexes
            openai_key: OpenAI API key
            default_fps: Default FPS to assume for frame sequences
        """
        self.frames_root_dir = Path(frames_root_dir)
        self.output_dir = Path(output_dir)
        self.default_fps = default_fps
        
        # Setup environment
        os.environ["OPENAI_API_KEY"] = openai_key
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize VideoRAG
        self.logger.info("Initializing VideoRAG framework...")
        self.videorag = VideoRAG(
            llm=openai_4o_mini_config,
            working_dir=str(self.output_dir)
        )
        
        # Track processed sequences
        self.processed_sequences = []
        self.frame_clips = []
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'frame_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FrameProcessor')
    
    def discover_frame_folders(self) -> List[Path]:
        """
        Discover all folders containing frame sequences
        
        Returns:
            List of folder paths containing frames
        """
        frame_folders = []
        
        for item in self.frames_root_dir.iterdir():
            if item.is_dir():
                # Check if folder contains image files
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_files.extend(item.glob(f'*{ext}'))
                    image_files.extend(item.glob(f'*{ext.upper()}'))
                
                if image_files:
                    frame_folders.append(item)
        
        self.logger.info(f"Discovered {len(frame_folders)} frame sequence folders")
        return frame_folders
    
    def process_frame_sequence(self, frames_folder: Path, 
                             fps: Optional[float] = None,
                             audio_file: Optional[str] = None) -> Dict:
        """
        Process a single frame sequence folder
        
        Args:
            frames_folder: Path to folder containing frames
            fps: FPS to use (default: self.default_fps)
            audio_file: Optional audio file path
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing frame sequence: {frames_folder.name}")
            
            # Create frame sequence clip
            fps = fps or self.default_fps
            frame_clip = FrameSequenceClip(
                frames_folder=str(frames_folder),
                fps=fps,
                audio_file=audio_file
            )
            
            # Get metadata
            metadata = frame_clip.get_metadata()
            
            # Store clip for later processing
            self.frame_clips.append(frame_clip)
            
            result = {
                'status': 'success',
                'folder_name': frames_folder.name,
                'metadata': metadata,
                'processing_time': time.time() - start_time,
                'temp_video_path': frame_clip.get_video_path()
            }
            
            self.processed_sequences.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {frames_folder.name}: {e}")
            return {
                'status': 'failed',
                'folder_name': frames_folder.name,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def build_videorag_index(self):
        """
        Build VideoRAG index from processed frame sequences
        """
        self.logger.info("Building VideoRAG index from frame sequences...")
        
        # Collect all temporary video paths
        video_paths = []
        for result in self.processed_sequences:
            if result['status'] == 'success':
                video_paths.append(result['temp_video_path'])
        
        if not video_paths:
            raise ValueError("No successfully processed frame sequences found!")
        
        # Load VideoRAG models
        self.logger.info("Loading VideoRAG models...")
        self.videorag.load_caption_model(debug=False)
        
        # Insert videos into VideoRAG
        self.logger.info(f"Inserting {len(video_paths)} videos into VideoRAG...")
        self.videorag.insert_video(video_path_list=video_paths)
        
        self.logger.info("VideoRAG index built successfully!")
    
    def query_frame_sequences(self, query: str, **kwargs) -> str:
        """
        Query the processed frame sequences
        
        Args:
            query: Search query
            **kwargs: Additional parameters for VideoRAG query
            
        Returns:
            Query response
        """
        param = QueryParam(mode="videorag")
        param.wo_reference = kwargs.get('wo_reference', True)
        
        response = self.videorag.query(query=query, param=param)
        return response
    
    def save_processing_manifest(self):
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
        
        self.logger.info(f"Processing manifest saved to {manifest_path}")
    
    def cleanup_temp_files(self):
        """Clean up temporary video files"""
        self.logger.info("Cleaning up temporary files...")
        
        for clip in self.frame_clips:
            clip.cleanup()
        
        # Remove temp directory if empty
        temp_dir = Path("./temp_videos")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    
    def run_full_processing(self):
        """
        Run complete frame sequence processing pipeline
        """
        self.logger.info("=== Starting Frame Sequence Processing ===")
        
        try:
            # Step 1: Discover frame folders
            frame_folders = self.discover_frame_folders()
            
            if not frame_folders:
                self.logger.error("No frame sequence folders found!")
                return False
            
            # Step 2: Process each frame sequence
            self.logger.info(f"Processing {len(frame_folders)} frame sequences...")
            
            for folder in tqdm(frame_folders, desc="Processing sequences"):
                self.process_frame_sequence(folder)
            
            # Step 3: Build VideoRAG index
            self.build_videorag_index()
            
            # Step 4: Save manifest
            self.save_processing_manifest()
            
            # Summary
            successful = sum(1 for r in self.processed_sequences if r['status'] == 'success')
            self.logger.info(f"=== Processing Complete ===")
            self.logger.info(f"Successfully processed: {successful}/{len(frame_folders)} sequences")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error: {e}")
            return False


def main():
    """Main entry point for frame sequence processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VideoRAG Frame Sequence Processor')
    parser.add_argument('--frames_dir', required=True, help='Root directory containing frame sequence folders')
    parser.add_argument('--output_dir', required=True, help='Output directory for VideoRAG indexes')
    parser.add_argument('--openai_key', required=True, help='OpenAI API key')
    parser.add_argument('--fps', type=float, default=30.0, help='Default FPS for frame sequences')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files after processing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoRAGFrameProcessor(
        frames_root_dir=args.frames_dir,
        output_dir=args.output_dir,
        openai_key=args.openai_key,
        default_fps=args.fps
    )
    
    # Run processing
    success = processor.run_full_processing()
    
    # Clean up if requested
    if args.cleanup:
        processor.cleanup_temp_files()
    
    return 0 if success else 1


# Example usage functions
def test_frame_sequence_query():
    """Test querying frame sequences"""
    # Initialize processor (assuming already processed)
    processor = VideoRAGFrameProcessor(
        frames_root_dir="./competition_data",
        output_dir="./frame_index",
        openai_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    # Test queries
    test_queries = [
        "Show me scenes with people walking",
        "Find outdoor scenes with trees",
        "Locate scenes with vehicles",
        "Where do people interact with objects?",
        "Find scenes with specific actions"
    ]
    
    print("🧪 Testing Frame Sequence Queries")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        start_time = time.time()
        response = processor.query_frame_sequences(query)
        query_time = time.time() - start_time
        
        print(f"⏱️  Response time: {query_time:.2f}s")
        print(f"📝 Response: {str(response)[:100]}...")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    exit(main())