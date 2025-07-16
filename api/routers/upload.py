from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from api.models.video_models import (
    VideoUploadResponse, VideoProcessRequest, VideoProcessResponse,
    VideoQueryRequest, VideoQueryResponse, VideoInfo, VideoListResponse,
    VideoDeleteResponse
)
from api.services.video_service import VideoService
from api.middleware.auth import get_current_user
from typing import Optional, List
import logging

router = APIRouter()
video_service = VideoService()

@router.post("/video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    openai_api_key: Optional[str] = Form(None),
    auto_process: Optional[bool] = Form(False),
    user=Depends(get_current_user)
):
    """
    Upload video file và tùy chọn auto-process với VideoRAG
    """
    # Kiểm tra file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # Kiểm tra file size (giới hạn 500MB)
    max_size = 500 * 1024 * 1024  # 500MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 500MB)")
    
    # Reset file pointer
    await file.seek(0)
    
    try:
        # Upload video
        result = await video_service.upload_video(file, user)
        
        # Auto process nếu được yêu cầu
        if auto_process and openai_api_key:
            try:
                await video_service.process_video(result["file_id"], openai_api_key)
                result["message"] = "Video uploaded and processed successfully"
            except Exception as e:
                logging.warning(f"Auto-processing failed: {e}")
                result["message"] = f"Video uploaded but processing failed: {str(e)}"
        
        return VideoUploadResponse(**result)
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video/{file_id}/process", response_model=VideoProcessResponse)
async def process_video(
    file_id: str,
    request: VideoProcessRequest,
    user=Depends(get_current_user)
):
    """
    Process video với VideoRAG để tạo index
    """
    try:
        result = await video_service.process_video(file_id, request.openai_api_key)
        return VideoProcessResponse(**result)
    except Exception as e:
        logging.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video/{file_id}/query", response_model=VideoQueryResponse)
async def query_video(
    file_id: str,
    request: VideoQueryRequest,
    user=Depends(get_current_user)
):
    """
    Query video content với VideoRAG
    """
    try:
        result = await video_service.query_video(file_id, request.query, request.openai_api_key)
        return VideoQueryResponse(**result)
    except Exception as e:
        logging.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{file_id}", response_model=VideoInfo)
async def get_video_info(
    file_id: str,
    user=Depends(get_current_user)
):
    """
    Lấy thông tin video
    """
    try:
        result = await video_service.get_video_info(file_id)
        return VideoInfo(**result)
    except Exception as e:
        logging.error(f"Get video info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos", response_model=VideoListResponse)
async def list_videos(user=Depends(get_current_user)):
    """
    Liệt kê tất cả video của user
    """
    try:
        videos = await video_service.list_videos(user)
        return VideoListResponse(videos=videos, total=len(videos))
    except Exception as e:
        logging.error(f"List videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/video/{file_id}", response_model=VideoDeleteResponse)
async def delete_video(
    file_id: str,
    user=Depends(get_current_user)
):
    """
    Xóa video
    """
    try:
        result = await video_service.delete_video(file_id, user)
        return VideoDeleteResponse(**result)
    except Exception as e:
        logging.error(f"Delete video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Frame sequence endpoints
@router.post("/frame-sequence")
async def upload_frame_sequence(
    folder_name: str = Form(...),
    fps: Optional[float] = Form(30.0),
    auto_process: Optional[bool] = Form(False),
    frame_files: List[UploadFile] = File(...),
    user=Depends(get_current_user)
):
    """
    Upload frame sequence for processing
    """
    try:
        # Validate frame files
        if not frame_files:
            raise HTTPException(status_code=400, detail="No frame files provided")
        
        # Check file types
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for frame_file in frame_files:
            if not any(frame_file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {frame_file.filename}. Allowed: {allowed_extensions}"
                )
        
        # Upload frame sequence
        result = await video_service.upload_frame_sequence(
            folder_name=folder_name,
            frame_files=frame_files,
            user_id=user
        )
        
        # Auto process if requested
        if auto_process and result['status'] == 'success':
            try:
                process_result = await video_service.process_frame_sequence(
                    folder_name=folder_name,
                    fps=fps
                )
                result["process_result"] = process_result
                result["message"] = "Frame sequence uploaded and processed successfully"
            except Exception as e:
                logging.warning(f"Auto-processing failed: {e}")
                result["message"] = f"Frame sequence uploaded but processing failed: {str(e)}"
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Upload frame sequence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/frame-sequence/{folder_name}/process")
async def process_frame_sequence(
    folder_name: str,
    fps: Optional[float] = Form(30.0),
    user=Depends(get_current_user)
):
    """
    Process frame sequence with VideoRAG
    """
    try:
        result = await video_service.process_frame_sequence(
            folder_name=folder_name,
            fps=fps
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Process frame sequence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/frame-sequence/query")
async def query_frame_sequence(
    query: str = Form(...),
    fast_mode: Optional[bool] = Form(True),
    user=Depends(get_current_user)
):
    """
    Query frame sequences
    """
    try:
        result = await video_service.query_frame_sequence(
            query=query,
            fast_mode=fast_mode
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Query frame sequence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/frame-sequence/build-index")
async def build_frame_index(
    user=Depends(get_current_user)
):
    """
    Build VideoRAG index from frame sequences
    """
    try:
        result = await video_service.build_frame_index()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Build frame index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/frame-sequence/status")
async def get_frame_processor_status(
    user=Depends(get_current_user)
):
    """
    Get frame processor status
    """
    try:
        result = await video_service.get_frame_processor_status()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Get frame processor status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/frame-sequence/cleanup")
async def cleanup_temp_files(
    user=Depends(get_current_user)
):
    """
    Clean up temporary files
    """
    try:
        result = await video_service.cleanup_temp_files()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Cleanup temp files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "upload"}
