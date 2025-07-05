# api/routers/video.py
"""
Video Router - API endpoints cho video processing system
Dev2: API Integration & Routing - chỉ tích hợp, không viết AI logic
Current: 2025-07-03 11:52:35 UTC, User: xthanh1910
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import uuid
import time

# Import agents từ Dev1 (người viết AI logic)
from ..agents_manager import agents_manager
from agents.orchestrator.preprocessing_orchestrator import PreprocessingConfig

# Import models cho API (Dev2 tạo đơn giản để frontend Dev4 sử dụng)
from ..models.video_models import VideoUploadResponse, VideoProcessRequest
from ..models.user_models import User

# Import services để connect với database Dev3
from ..services.video_service import VideoService
from ..middleware.auth import get_current_user

# Khởi tạo service để làm việc với database
video_service = VideoService()
router = APIRouter(prefix="/video", tags=["video"])

# Cấu hình upload (Dev2 setup đơn giản)
UPLOAD_DIR = "uploads/videos"
MAX_FILE_SIZE = 500 * 1024 * 1024   # 500MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Tạo thư mục upload nếu chưa có
os.makedirs(UPLOAD_DIR, exist_ok=True)

#==========================================================================================================================================
# VIDEO UPLOAD ENDPOINTS
#==========================================================================================================================================

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),  # File upload từ frontend Dev4
    title: str = Form(""),         # Tiêu đề video
    description: str = Form(""),   # Mô tả video
    auto_process: bool = Form(True),  # Có tự động xử lý không
    current_user: User = Depends(get_current_user)  # User hiện tại
):
    """
    Upload video file và lưu thông tin
    Dev2 chỉ làm: nhận file -> lưu file -> gọi agents Dev1 -> lưu database Dev3
    """
    try:
        # Bước 1: Kiểm tra file có hợp lệ không
        if not file.filename:
            raise HTTPException(status_code=400, detail="Không có file được chọn")

        # Kiểm tra đuôi file có được phép không
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Bước 2: Tạo tên file mới để tránh trùng lặp
        video_id = str(uuid.uuid4())  # Tạo ID duy nhất cho video
        new_filename = f"{video_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, new_filename)

        # Bước 3: Lưu file vào server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  # Copy file từ request vào server

        # Bước 4: Lưu thông tin video vào database thông qua Dev3's service
        video_metadata = {
            "video_id": video_id,
            "original_filename": file.filename,
            "file_path": file_path,
            "title": title or file.filename,
            "description": description,
            "uploaded_by": current_user.user_id,
            "uploaded_at": time.time(),
            "status": "uploaded",  # Trạng thái: đã upload
            "file_size": os.path.getsize(file_path)
        }

        # Gọi service Dev3 để lưu vào database
        await video_service.save_video_metadata(video_metadata)

        # Bước 5: Bắt đầu xử lý video nếu user chọn auto_process
        job_id = None
        if auto_process:
            # Tạo job ID để track tiến trình
            job_id = f"job_{video_id}_{int(time.time())}"

            # Gọi service để bắt đầu processing (sẽ gọi agents Dev1)
            await video_service.start_video_processing(
                job_id=job_id,
                video_id=video_id,
                video_path=file_path,
                user_id=current_user.user_id
            )

        # Bước 6: Trả về response cho frontend Dev4
        return VideoUploadResponse(
            success=True,
            video_id=video_id,
            job_id=job_id,
            filename=file.filename,
            file_path=file_path,
            status="processing" if auto_process else "uploaded",
            message="Video uploaded successfully" + (" and processing started" if auto_process else "")
        )

    except Exception as e:
        # Nếu có lỗi, xóa file đã upload (cleanup)
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload/multiple")
async def upload_multiple_videos(
    files: List[UploadFile] = File(...),  # Nhiều files từ frontend
    auto_process: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    """
    Upload nhiều video cùng lúc (batch upload)
    """
    try:
        # Giới hạn số lượng file để tránh quá tải server
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Tối đa 10 files mỗi lần upload")

        upload_results = []  # Danh sách kết quả upload
        failed_files = []    # Danh sách files upload thất bại

        # Xử lý từng file một
        for file in files:
            try:
                # Tương tự logic upload single file
                if not file.filename:
                    failed_files.append({"filename": "unknown", "error": "No filename"})
                    continue

                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in ALLOWED_EXTENSIONS:
                    failed_files.append({"filename": file.filename, "error": "Unsupported format"})
                    continue

                # Lưu file
                video_id = str(uuid.uuid4())
                new_filename = f"{video_id}_{file.filename}"
                file_path = os.path.join(UPLOAD_DIR, new_filename)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Lưu metadata vào database
                video_metadata = {
                    "video_id": video_id,
                    "original_filename": file.filename,
                    "file_path": file_path,
                    "title": file.filename,
                    "uploaded_by": current_user.user_id,
                    "uploaded_at": time.time(),
                    "status": "uploaded",
                    "file_size": os.path.getsize(file_path)
                }

                await video_service.save_video_metadata(video_metadata)

                # Thêm vào danh sách thành công
                upload_results.append({
                    "video_id": video_id,
                    "filename": file.filename,
                    "status": "uploaded"
                })

            except Exception as e:
                # Nếu file này lỗi, tiếp tục với file khác
                failed_files.append({"filename": file.filename, "error": str(e)})

        return {
            "total_files": len(files),
            "successful": len(upload_results),
            "failed": len(failed_files),
            "upload_results": upload_results,
            "failed_files": failed_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

#==========================================================================================================================================
# VIDEO PROCESSING ENDPOINTS
#==========================================================================================================================================

@router.post("/process/{video_id}")
async def process_video_by_id(
    video_id: str,
    background_tasks: BackgroundTasks,  # Để chạy processing ở background
    current_user: User = Depends(get_current_user)
):
    """
    Bắt đầu xử lý video theo ID
    Dev2 gọi agents Dev1
    """
    try:
        # Bước 1: Lấy thông tin video từ database thông qua Dev3
        video_metadata = await video_service.get_video_by_id(video_id)
        if not video_metadata:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Bước 2: Kiểm tra quyền - chỉ owner hoặc admin mới được xử lý
        if (video_metadata["uploaded_by"] != current_user.user_id and
            not current_user.is_admin()):
            raise HTTPException(status_code=403, detail="Không có quyền truy cập video này")

        # Bước 3: Tạo job để track tiến trình
        job_id = f"job_{video_id}_{int(time.time())}"

        # Bước 4: Bắt đầu processing bằng cách gọi service
        # Service sẽ gọi PreprocessingOrchestrator từ Dev1
        await video_service.start_video_processing(
            job_id=job_id,
            video_id=video_id,
            video_path=video_metadata["file_path"],
            user_id=current_user.user_id
        )

        return {
            "success": True,
            "job_id": job_id,
            "video_id": video_id,
            "message": "Video processing started",
            "status": "processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/process")
async def process_video_legacy(request: VideoProcessRequest):
    """
    Legacy endpoint - xử lý video bằng path (tương thích với code cũ)
    """
    try:
        # Lấy preprocessing orchestrator từ Dev1
        orchestrator = agents_manager.get_preprocessing_orchestrator()

        # Tạo config cho processing nếu có
        config = None
        if request.config:
            config = PreprocessingConfig(**request.config)

        # Gọi orchestrator để xử lý video (Dev1's AI logic)
        result = orchestrator.process_video(
            video_path=request.video_path,
            config=config
        )

        # Kiểm tra kết quả từ agents
        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        # Extract thông tin từ kết quả processing
        pipeline_data = result.result
        return {
            "status": "success",
            "pipeline_id": pipeline_data.get("pipeline_id"),
            "video_id": pipeline_data.get("video_id"),
            "processing_time": result.execution_time,
            "stages_completed": len(pipeline_data.get("stage_executions", [])),
            "success_rate": pipeline_data.get("success_rate", 0),
            "quality_score": pipeline_data.get("overall_quality_score", 0),
            "search_ready": pipeline_data.get("indexing_result") is not None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

#==========================================================================================================================================
# VIDEO INFORMATION ENDPOINTS
#==========================================================================================================================================

@router.get("/{video_id}")
async def get_video_details(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Lấy thông tin chi tiết của video
    """
    try:
        # Lấy thông tin từ database qua Dev3's service
        video_metadata = await video_service.get_video_by_id(video_id)
        if not video_metadata:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Kiểm tra quyền truy cập (private videos)
        if (video_metadata.get("privacy") == "private" and
            video_metadata["uploaded_by"] != current_user.user_id and
            not current_user.is_admin()):
            raise HTTPException(status_code=403, detail="Không có quyền truy cập video này")

        return video_metadata

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video: {str(e)}")

@router.get("/")
async def list_videos(
    page: int = 1,           # Trang hiện tại (phân trang)
    limit: int = 20,         # Số videos mỗi trang
    status: Optional[str] = None,    # Lọc theo trạng thái
    user_only: bool = False, # Chỉ lấy videos của user hiện tại
    current_user: User = Depends(get_current_user)
):
    """
    Lấy danh sách videos với phân trang và filter
    """
    try:
        # Tạo filter parameters
        filters = {
            "page": page,
            "limit": limit,
        }

        if status:
            filters["status"] = status

        if user_only:
            filters["uploaded_by"] = current_user.user_id

        # Lấy danh sách từ database qua Dev3's service
        videos = await video_service.get_videos_list(filters)

        return videos

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

#==========================================================================================================================================
# JOB STATUS ENDPOINTS
#==========================================================================================================================================

@router.get("/job/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Lấy trạng thái của job processing
    Để frontend Dev4 có thể hiển thị progress bar
    """
    try:
        # Lấy trạng thái job từ database qua Dev3's service
        job_status = await video_service.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail="Job không tồn tại")

        return {
            "job_id": job_id,
            "status": job_status.get("status", "unknown"),  # pending, processing, completed, failed
            "progress": job_status.get("progress", 0),      # % hoàn thành
            "current_stage": job_status.get("current_stage", ""),
            "started_at": job_status.get("started_at"),
            "completed_at": job_status.get("completed_at"),
            "error_message": job_status.get("error_message"),
            "result": job_status.get("result")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.delete("/job/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Hủy job đang chạy (chỉ admin hoặc owner)
    """
    try:
        # Kiểm tra quyền và hủy job
        success = await video_service.cancel_job(job_id, current_user.user_id)

        if not success:
            raise HTTPException(status_code=403, detail="Không có quyền hủy job này")

        return {"success": True, "message": "Job đã được hủy"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

#==========================================================================================================================================
# VIDEO DOWNLOAD ENDPOINT
#==========================================================================================================================================

@router.get("/{video_id}/download")
async def download_video(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download video file gốc
    """
    try:
        # Lấy thông tin video
        video_metadata = await video_service.get_video_by_id(video_id)
        if not video_metadata:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Kiểm tra quyền
        if (video_metadata.get("privacy") == "private" and
            video_metadata["uploaded_by"] != current_user.user_id and
            not current_user.is_admin()):
            raise HTTPException(status_code=403, detail="Không có quyền download video này")

        # Kiểm tra file có tồn tại không
        file_path = video_metadata["file_path"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File video không tồn tại trên server")

        # Trả về file để download
        from fastapi.responses import FileResponse
        return FileResponse(
            path=file_path,
            filename=video_metadata["original_filename"],
            media_type='application/octet-stream'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

#==========================================================================================================================================
# HEALTH CHECK
#==========================================================================================================================================

@router.get("/health")
async def video_health():
    """
    Health check cho video system
    """
    try:
        # Test preprocessing orchestrator connection
        orchestrator = agents_manager.get_preprocessing_orchestrator()

        # Test upload directory
        upload_dir_ok = os.path.exists(UPLOAD_DIR) and os.path.isdir(UPLOAD_DIR)

        # Test disk space (đơn giản)
        import shutil
        disk_usage = shutil.disk_usage(UPLOAD_DIR)
        free_space_gb = disk_usage.free / (1024**3)

        return {
            "status": "healthy",
            "preprocessing_orchestrator": "ready" if orchestrator else "unavailable",
            "upload_directory": "ok" if upload_dir_ok else "error",
            "free_space_gb": round(free_space_gb, 2),
            "max_file_size_mb": MAX_FILE_SIZE // (1024**2),
            "allowed_formats": list(ALLOWED_EXTENSIONS),
            "endpoints": [
                "/video/upload",
                "/video/upload/multiple",
                "/video/process/{video_id}",
                "/video/{video_id}",
                "/video/job/{job_id}/status"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }