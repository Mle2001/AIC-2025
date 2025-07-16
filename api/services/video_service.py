import os
import sys
import json
import uuid
import asyncio
import aiofiles
from typing import Optional, List, Dict, Any
from fastapi import UploadFile, HTTPException
from api.config import get_openai_api_key, settings
import logging

# Use compatibility layer for VideoRAG imports
try:
    from .videorag_compat import (
        VIDEORAG_AVAILABLE, 
        VideoRAG, 
        QueryParam, 
        LLMConfig, 
        openai_embedding, 
        gpt_4o_mini_complete
    )
except ImportError:
    logging.warning("VideoRAG compatibility layer failed")
    VIDEORAG_AVAILABLE = False

# Import frame processor service for integration
# from api.services.frame_processor_service import frame_processor_service

class VideoService:
    def __init__(self):
        self.upload_dir = settings.upload_dir + "/videos"
        self.processed_videos = {}
        self.videorag_instances = {}
        
        # Tạo thư mục upload nếu chưa tồn tại
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def _create_llm_config(self, openai_api_key: Optional[str] = None) -> 'LLMConfig':
        """Tạo config cho VideoRAG với OpenAI API key"""
        if not VIDEORAG_AVAILABLE:
            raise HTTPException(status_code=500, detail="VideoRAG not available")
        
        # Sử dụng API key từ parameter hoặc config
        api_key = openai_api_key or get_openai_api_key()
        
        return LLMConfig(
            embedding_func_raw=openai_embedding,
            embedding_model_name=settings.openai_embedding_model,
            embedding_dim=1536,
            embedding_max_token_size=8192,
            embedding_batch_num=32,
            embedding_func_max_async=16,
            query_better_than_threshold=0.2,
            best_model_func_raw=gpt_4o_mini_complete,
            best_model_name=settings.openai_model,
            best_model_max_token_size=32768,
            best_model_max_async=16,
            cheap_model_func_raw=gpt_4o_mini_complete,
            cheap_model_name=settings.openai_model,
            cheap_model_max_token_size=32768,
            cheap_model_max_async=16
        )
    
    async def upload_video(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """Upload và lưu video file"""
        try:
            # Tạo unique filename
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
            filename = f"{file_id}.{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Lưu file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Lưu metadata
            video_info = {
                "file_id": file_id,
                "original_name": file.filename,
                "file_path": file_path,
                "file_size": len(content),
                "user_id": user_id,
                "status": "uploaded",
                "processed": False
            }
            
            self.processed_videos[file_id] = video_info
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "size": len(content),
                "status": "uploaded"
            }
            
        except Exception as e:
            logging.error(f"Error uploading video: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def process_video(self, file_id: str, openai_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Xử lý video với VideoRAG"""
        if not VIDEORAG_AVAILABLE:
            raise HTTPException(status_code=500, detail="VideoRAG not available")
        
        if file_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = self.processed_videos[file_id]
        
        try:
            # Set OpenAI API key
            api_key = openai_api_key or get_openai_api_key()
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Tạo LLM config
            llm_config = self._create_llm_config(api_key)
            
            # Khởi tạo VideoRAG với LLM config
            videorag = VideoRAG(
                working_dir=f"./videorag_cache_{file_id}",
                llm=llm_config
            )
            
            # Process video
            video_path = video_info["file_path"]
            
            # Insert video into VideoRAG (indexing)
            try:
                await asyncio.to_thread(videorag.insert_video, video_path_list=[video_path])
            except RuntimeError as e:
                # Handle VideoRAG processing errors gracefully
                error_msg = str(e)
                if "bitsandbytes" in error_msg:
                    error_msg = "Video processing failed: Missing dependencies for model quantization. Please install bitsandbytes package."
                elif "MiniCPM" in error_msg or "caption" in error_msg:
                    error_msg = "Video processing failed: Vision model error. Using fallback processing."
                else:
                    error_msg = f"Video processing failed: {error_msg}"
                
                logging.error(f"VideoRAG processing error: {error_msg}")
                video_info["status"] = "error"
                video_info["error"] = error_msg
                raise HTTPException(status_code=500, detail=error_msg)
            except Exception as e:
                error_msg = f"Video processing failed: {str(e)}"
                logging.error(f"VideoRAG unexpected error: {error_msg}")
                video_info["status"] = "error"
                video_info["error"] = error_msg
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Lưu VideoRAG instance
            self.videorag_instances[file_id] = videorag
            
            # Cập nhật status
            video_info["status"] = "processed"
            video_info["processed"] = True
            
            return {
                "file_id": file_id,
                "status": "processed",
                "message": "Video processed successfully"
            }
            
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            video_info["status"] = "error"
            video_info["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    async def query_video(self, file_id: str, query: str, openai_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Query video với VideoRAG"""
        if not VIDEORAG_AVAILABLE:
            raise HTTPException(status_code=500, detail="VideoRAG not available")
        
        if file_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if file_id not in self.videorag_instances:
            raise HTTPException(status_code=400, detail="Video not processed yet")
        
        try:
            # Set OpenAI API key
            api_key = openai_api_key or get_openai_api_key()
            os.environ["OPENAI_API_KEY"] = api_key
            
            videorag = self.videorag_instances[file_id]
            
            # Tạo query param
            query_param = QueryParam(query=query)
            
            # Thực hiện query
            result = await asyncio.to_thread(videorag.query, query_param)
            
            return {
                "file_id": file_id,
                "query": query,
                "response": result.response if hasattr(result, 'response') else str(result),
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"Error querying video: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    async def get_video_info(self, file_id: str) -> Dict[str, Any]:
        """Lấy thông tin video"""
        if file_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = self.processed_videos[file_id].copy()
        # Xóa file_path khỏi response vì lý do bảo mật
        video_info.pop("file_path", None)
        
        return video_info
    
    async def list_videos(self, user_id: str) -> List[Dict[str, Any]]:
        """Liệt kê tất cả video của user"""
        user_videos = []
        for file_id, video_info in self.processed_videos.items():
            if video_info["user_id"] == user_id:
                info = video_info.copy()
                info.pop("file_path", None)  # Bảo mật
                user_videos.append(info)
        
        return user_videos
    
    async def delete_video(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """Xóa video"""
        if file_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = self.processed_videos[file_id]
        
        # Kiểm tra quyền sở hữu
        if video_info["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this video")
        
        try:
            # Xóa file
            if os.path.exists(video_info["file_path"]):
                os.remove(video_info["file_path"])
            
            # Xóa VideoRAG instance
            if file_id in self.videorag_instances:
                del self.videorag_instances[file_id]
            
            # Xóa metadata
            del self.processed_videos[file_id]
            
            return {
                "file_id": file_id,
                "status": "deleted",
                "message": "Video deleted successfully"
            }
            
        except Exception as e:
            logging.error(f"Error deleting video: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
    # Frame sequence integration methods
    async def upload_frame_sequence(self, folder_name: str, frame_files: List[UploadFile], user_id: str) -> Dict[str, Any]:
        """Upload frame sequence and delegate to frame processor"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            # Delegate to frame processor service
            result = await frame_processor_service.upload_frame_sequence(
                folder_name=folder_name,
                frame_files=frame_files
            )
            
            # Store user association
            if result['status'] == 'success':
                frame_info = {
                    "folder_name": folder_name,
                    "user_id": user_id,
                    "type": "frame_sequence",
                    "status": "uploaded",
                    "processed": False,
                    "frame_count": result['frame_count'],
                    "folder_path": result['folder_path']
                }
                
                self.processed_videos[f"frame_{folder_name}"] = frame_info
            
            return result
            
        except Exception as e:
            logging.error(f"Error uploading frame sequence: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def process_frame_sequence(self, folder_name: str, fps: Optional[float] = None) -> Dict[str, Any]:
        """Process frame sequence using frame processor"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            # Delegate to frame processor service
            result = await frame_processor_service.process_frame_sequence(
                folder_name=folder_name,
                fps=fps
            )
            
            # Update status
            frame_key = f"frame_{folder_name}"
            if frame_key in self.processed_videos:
                if result['status'] == 'success':
                    self.processed_videos[frame_key]["status"] = "processed"
                    self.processed_videos[frame_key]["processed"] = True
                else:
                    self.processed_videos[frame_key]["status"] = "error"
                    self.processed_videos[frame_key]["error"] = result.get('error', 'Processing failed')
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing frame sequence: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    async def query_frame_sequence(self, query: str, fast_mode: bool = True) -> Dict[str, Any]:
        """Query frame sequences using frame processor"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            # Delegate to frame processor service
            result = await frame_processor_service.query_frame_sequences(
                query=query,
                fast_mode=fast_mode
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error querying frame sequences: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    async def get_frame_processor_status(self) -> Dict[str, Any]:
        """Get frame processor status"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            return await frame_processor_service.get_processing_status()
        except Exception as e:
            logging.error(f"Error getting frame processor status: {e}")
            raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
    
    async def build_frame_index(self) -> Dict[str, Any]:
        """Build VideoRAG index from frame sequences"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            return await frame_processor_service.build_videorag_index()
        except Exception as e:
            logging.error(f"Error building frame index: {e}")
            raise HTTPException(status_code=500, detail=f"Index building failed: {str(e)}")
    
    async def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            await frame_processor_service.cleanup_temp_files()
            return {
                "status": "success",
                "message": "Temporary files cleaned up successfully"
            }
        except Exception as e:
            logging.error(f"Error cleaning up temp files: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
    def is_frame_sequence_ready(self) -> bool:
        """Check if frame sequence processing is ready"""
        try:
            # Lazy import to avoid circular import
            from api.services.frame_processor_service import frame_processor_service
            
            status = asyncio.run(self.get_frame_processor_status())
            return status['status'] == 'ready'
        except:
            return False


# Global service instance
video_service = VideoService()