# api/models/video_models.py
"""
Video Models - Pydantic models cho video processing system
Dev2: API Data Models - định nghĩa structure cho video requests/responses
Current: 2025-07-03 14:16:45 UTC, User: xthanh1910
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import os
from pathlib import Path

# ================================
# ENUMS CHO VIDEO SYSTEM
# ================================

class VideoStatus(str, Enum):
    """
    Trạng thái của video trong hệ thống
    """
    UPLOADED = "uploaded"           # Vừa upload xong
    PROCESSING = "processing"       # Đang xử lý
    PROCESSED = "processed"         # Đã xử lý xong
    FAILED = "failed"              # Xử lý thất bại
    CANCELLED = "cancelled"         # Bị hủy
    ARCHIVED = "archived"          # Đã archive
    DELETED = "deleted"            # Đã xóa

class VideoType(str, Enum):
    """
    Loại video theo content
    """
    EDUCATIONAL = "educational"     # Video giáo dục
    ENTERTAINMENT = "entertainment" # Video giải trí
    TUTORIAL = "tutorial"          # Video hướng dẫn
    DOCUMENTARY = "documentary"     # Video tài liệu
    LECTURE = "lecture"            # Video bài giảng
    COOKING = "cooking"            # Video nấu ăn
    TRAVEL = "travel"              # Video du lịch
    TECH = "tech"                  # Video công nghệ
    NEWS = "news"                  # Video tin tức
    OTHER = "other"                # Khác

class ProcessingStage(str, Enum):
    """
    Các giai đoạn processing video
    """
    INITIALIZING = "initializing"           # Khởi tạo
    VIDEO_ANALYSIS = "video_analysis"       # Phân tích video
    AUDIO_EXTRACTION = "audio_extraction"   # Trích xuất audio
    FRAME_EXTRACTION = "frame_extraction"   # Trích xuất frames
    SPEECH_TO_TEXT = "speech_to_text"       # Chuyển speech thành text
    FEATURE_EXTRACTION = "feature_extraction" # Trích xuất features
    CONTENT_ANALYSIS = "content_analysis"    # Phân tích nội dung
    KNOWLEDGE_GRAPH = "knowledge_graph"      # Tạo knowledge graph
    INDEXING = "indexing"                   # Đánh index
    FINISHED = "finished"                   # Hoàn thành

class JobStatus(str, Enum):
    """
    Trạng thái của processing job
    """
    PENDING = "pending"             # Đang chờ
    RUNNING = "running"             # Đang chạy
    COMPLETED = "completed"         # Hoàn thành
    FAILED = "failed"              # Thất bại
    CANCELLED = "cancelled"         # Bị hủy
    STOPPED = "stopped"            # Dừng khẩn cấp

class FileType(str, Enum):
    """
    Loại file được upload
    """
    VIDEO = "video"                # Video files
    DOCUMENT = "document"          # Document files
    IMAGE = "image"               # Image files
    TEMP = "temp"                 # Temporary files

# ================================
# VIDEO UPLOAD MODELS
# ================================

class VideoUploadRequest(BaseModel):
    """
    Request cho video upload (from form data)
    """
    title: Optional[str] = Field(
        None,
        max_length=200,
        description="Tiêu đề video"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Mô tả video"
    )
    tags: Optional[str] = Field(
        None,
        description="Tags cách nhau bởi dấu phẩy"
    )
    video_type: VideoType = Field(
        VideoType.OTHER,
        description="Loại video"
    )
    privacy: str = Field(
        "public",
        regex="^(public|private|unlisted)$",
        description="Mức độ riêng tư"
    )
    auto_process: bool = Field(
        True,
        description="Tự động xử lý video sau khi upload"
    )

    @validator('title')
    def validate_title(cls, v):
        if v:
            v = v.strip()
            if len(v) < 3:
                raise ValueError('Tiêu đề phải có ít nhất 3 ký tự')
        return v

    @validator('tags')
    def validate_tags(cls, v):
        if v:
            # Clean up tags
            tags = [tag.strip() for tag in v.split(',') if tag.strip()]
            if len(tags) > 10:
                raise ValueError('Tối đa 10 tags')
            return ','.join(tags)
        return v

    class Config:
        schema_extra = {
            "example": {
                "title": "Cách nấu phở bò truyền thống",
                "description": "Video hướng dẫn chi tiết cách nấu phở bò ngon như quán",
                "tags": "nấu ăn, phở bò, món việt, tutorial",
                "video_type": "cooking",
                "privacy": "public",
                "auto_process": True
            }
        }

class FileUploadResponse(BaseModel):
    """
    Response cho file upload
    """
    success: bool = Field(..., description="Upload có thành công không")
    file_id: str = Field(..., description="ID của file đã upload")
    job_id: Optional[str] = Field(None, description="ID của processing job")
    original_filename: str = Field(..., description="Tên file gốc")
    file_size: int = Field(..., description="Kích thước file (bytes)")
    file_type: FileType = Field(..., description="Loại file")
    upload_path: str = Field(..., description="Đường dẫn file trên server")
    status: VideoStatus = Field(..., description="Trạng thái hiện tại")
    warnings: List[str] = Field([], description="Cảnh báo nếu có")
    message: str = Field(..., description="Thông báo kết quả")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "file_id": "file_abc123",
                "job_id": "job_def456",
                "original_filename": "pho_recipe.mp4",
                "file_size": 157286400,
                "file_type": "video",
                "upload_path": "/uploads/videos/file_abc123_pho_recipe.mp4",
                "status": "processing",
                "warnings": [],
                "message": "Video uploaded successfully and processing started"
            }
        }

class BatchUploadResponse(BaseModel):
    """
    Response cho batch upload nhiều files
    """
    success: bool = Field(..., description="Batch upload có thành công không")
    batch_id: str = Field(..., description="ID của batch")
    batch_job_id: Optional[str] = Field(None, description="ID của batch processing job")
    total_files: int = Field(..., description="Tổng số files")
    successful_uploads: int = Field(..., description="Số files upload thành công")
    failed_uploads: int = Field(..., description="Số files upload thất bại")
    successful_files: List[Dict[str, Any]] = Field([], description="Danh sách files thành công")
    failed_files: List[Dict[str, Any]] = Field([], description="Danh sách files thất bại")
    total_size: int = Field(..., description="Tổng kích thước (bytes)")
    upload_duration_seconds: float = Field(..., description="Thời gian upload")
    message: str = Field(..., description="Thông báo kết quả")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "batch_id": "batch_123",
                "batch_job_id": "batch_job_456",
                "total_files": 5,
                "successful_uploads": 4,
                "failed_uploads": 1,
                "successful_files": [
                    {
                        "file_id": "file_1",
                        "filename": "video1.mp4",
                        "file_size": 100000000,
                        "status": "uploaded"
                    }
                ],
                "failed_files": [
                    {
                        "filename": "video_corrupt.avi",
                        "error": "File format not supported"
                    }
                ],
                "total_size": 500000000,
                "upload_duration_seconds": 45.2,
                "message": "Uploaded 4/5 files successfully"
            }
        }

# ================================
# VIDEO PROCESSING MODELS
# ================================

class VideoProcessRequest(BaseModel):
    """
    Request để bắt đầu video processing
    """
    video_path: str = Field(..., description="Đường dẫn tới video file")
    config: Optional[Dict[str, Any]] = Field(None, description="Cấu hình processing")
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Độ ưu tiên (1=thấp, 10=cao)"
    )
    callback_url: Optional[str] = Field(None, description="URL callback khi hoàn thành")

    @validator('video_path')
    def validate_video_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'Video file không tồn tại: {v}')

        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        file_ext = Path(v).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f'Định dạng file không được hỗ trợ: {file_ext}')

        return v

    class Config:
        schema_extra = {
            "example": {
                "video_path": "/uploads/videos/sample_video.mp4",
                "config": {
                    "extract_frames": True,
                    "frame_interval": 30,
                    "extract_audio": True,
                    "speech_to_text": True,
                    "language": "vi"
                },
                "priority": 7,
                "callback_url": "https://api.example.com/webhook/processing"
            }
        }

class ProcessingJobInfo(BaseModel):
    """
    Thông tin chi tiết về processing job
    """
    job_id: str = Field(..., description="ID của job")
    video_id: str = Field(..., description="ID của video")
    status: JobStatus = Field(..., description="Trạng thái job")
    progress: int = Field(..., ge=0, le=100, description="Tiến độ (%)")
    current_stage: ProcessingStage = Field(..., description="Giai đoạn hiện tại")
    started_at: Optional[datetime] = Field(None, description="Thời gian bắt đầu")
    completed_at: Optional[datetime] = Field(None, description="Thời gian hoàn thành")
    estimated_completion: Optional[str] = Field(None, description="Ước tính hoàn thành")
    error_message: Optional[str] = Field(None, description="Thông báo lỗi")
    result: Optional[Dict[str, Any]] = Field(None, description="Kết quả processing")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_123",
                "video_id": "video_456",
                "status": "running",
                "progress": 65,
                "current_stage": "feature_extraction",
                "started_at": "2025-07-03T14:16:45Z",
                "completed_at": None,
                "estimated_completion": "~5 minutes",
                "error_message": None,
                "result": None
            }
        }

class ProcessingResult(BaseModel):
    """
    Kết quả sau khi processing video
    """
    video_id: str = Field(..., description="ID của video")
    pipeline_id: str = Field(..., description="ID của processing pipeline")
    processing_time: float = Field(..., description="Thời gian xử lý (seconds)")
    stages_completed: int = Field(..., description="Số giai đoạn đã hoàn thành")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Tỷ lệ thành công")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Điểm chất lượng")
    search_ready: bool = Field(..., description="Sẵn sàng cho search chưa")

    # Extracted data
    extracted_text: Optional[str] = Field(None, description="Text đã trích xuất")
    key_topics: List[str] = Field([], description="Chủ đề chính")
    entities: List[str] = Field([], description="Entities được nhận diện")
    frames_extracted: int = Field(0, description="Số frames đã trích xuất")
    audio_duration: Optional[float] = Field(None, description="Thời lượng audio")

    # Analysis results
    content_categories: List[str] = Field([], description="Danh mục nội dung")
    sentiment_score: Optional[float] = Field(None, description="Điểm sentiment")
    complexity_level: Optional[str] = Field(None, description="Mức độ phức tạp")

    class Config:
        schema_extra = {
            "example": {
                "video_id": "video_456",
                "pipeline_id": "pipeline_789",
                "processing_time": 245.7,
                "stages_completed": 9,
                "success_rate": 0.94,
                "quality_score": 0.87,
                "search_ready": True,
                "extracted_text": "Hôm nay tôi sẽ hướng dẫn các bạn cách nấu phở bò...",
                "key_topics": ["nấu ăn", "phở bố", "món việt"],
                "entities": ["phở", "thịt bò", "hành tây", "gừng"],
                "frames_extracted": 150,
                "audio_duration": 600.5,
                "content_categories": ["cooking", "tutorial"],
                "sentiment_score": 0.8,
                "complexity_level": "intermediate"
            }
        }

# ================================
# VIDEO METADATA MODELS
# ================================

class VideoMetadata(BaseModel):
    """
    Metadata đầy đủ của video
    """
    video_id: str = Field(..., description="ID của video")
    original_filename: str = Field(..., description="Tên file gốc")
    title: str = Field(..., description="Tiêu đề video")
    description: Optional[str] = Field(None, description="Mô tả video")
    tags: List[str] = Field([], description="Tags của video")
    video_type: VideoType = Field(..., description="Loại video")

    # File info
    file_path: str = Field(..., description="Đường dẫn file")
    file_size: int = Field(..., description="Kích thước file")
    duration: Optional[float] = Field(None, description="Thời lượng (seconds)")
    resolution: Optional[str] = Field(None, description="Độ phân giải")
    format: Optional[str] = Field(None, description="Format video")

    # Status & ownership
    status: VideoStatus = Field(..., description="Trạng thái video")
    privacy: str = Field(..., description="Mức độ riêng tư")
    uploaded_by: str = Field(..., description="ID người upload")
    uploaded_at: datetime = Field(..., description="Thời gian upload")
    updated_at: datetime = Field(..., description="Thời gian cập nhật")

    # Processing info
    processing_attempts: int = Field(0, description="Số lần thử processing")
    last_processed_at: Optional[datetime] = Field(None, description="Lần xử lý cuối")
    processing_result: Optional[ProcessingResult] = Field(None, description="Kết quả processing")

    # Statistics
    view_count: int = Field(0, description="Số lượt xem")
    search_count: int = Field(0, description="Số lần được tìm thấy")
    rating: Optional[float] = Field(None, description="Đánh giá trung bình")

    class Config:
        schema_extra = {
            "example": {
                "video_id": "video_456",
                "original_filename": "pho_recipe.mp4",
                "title": "Cách nấu phở bò truyền thống",
                "description": "Video hướng dẫn chi tiết...",
                "tags": ["nấu ăn", "phở bò", "món việt"],
                "video_type": "cooking",
                "file_path": "/uploads/videos/video_456_pho_recipe.mp4",
                "file_size": 157286400,
                "duration": 600.5,
                "resolution": "1920x1080",
                "format": "mp4",
                "status": "processed",
                "privacy": "public",
                "uploaded_by": "user_123",
                "uploaded_at": "2025-07-03T14:16:45Z",
                "updated_at": "2025-07-03T14:20:30Z",
                "processing_attempts": 1,
                "last_processed_at": "2025-07-03T14:18:15Z",
                "view_count": 245,
                "search_count": 89,
                "rating": 4.5
            }
        }

class VideoSearchResult(BaseModel):
    """
    Kết quả tìm kiếm video
    """
    video_id: str = Field(..., description="ID của video")
    title: str = Field(..., description="Tiêu đề video")
    description: Optional[str] = Field(None, description="Mô tả ngắn")
    thumbnail_url: Optional[str] = Field(None, description="URL thumbnail")
    duration: Optional[float] = Field(None, description="Thời lượng")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Điểm relevance")

    # Matching info
    matched_segments: List[Dict[str, Any]] = Field([], description="Segments trùng khớp")
    highlighted_text: Optional[str] = Field(None, description="Text được highlight")
    matching_tags: List[str] = Field([], description="Tags trùng khớp")

    # Statistics
    view_count: int = Field(0, description="Số lượt xem")
    rating: Optional[float] = Field(None, description="Đánh giá")

    # Metadata
    video_type: VideoType = Field(..., description="Loại video")
    uploaded_at: datetime = Field(..., description="Ngày upload")

    class Config:
        schema_extra = {
            "example": {
                "video_id": "video_456",
                "title": "Cách nấu phở bò truyền thống",
                "description": "Video hướng dẫn chi tiết cách nấu phở bò ngon...",
                "thumbnail_url": "https://example.com/thumb_456.jpg",
                "duration": 600.5,
                "relevance_score": 0.92,
                "matched_segments": [
                    {
                        "start_time": 120,
                        "end_time": 180,
                        "text": "Bước đầu tiên là làm nước dùng phở",
                        "confidence": 0.95
                    }
                ],
                "highlighted_text": "nấu <em>phở bò</em> truyền thống",
                "matching_tags": ["nấu ăn", "phở bò"],
                "view_count": 245,
                "rating": 4.5,
                "video_type": "cooking",
                "uploaded_at": "2025-07-03T14:16:45Z"
            }
        }

# ================================
# VIDEO MANAGEMENT MODELS
# ================================

class VideoListRequest(BaseModel):
    """
    Request để lấy danh sách videos
    """
    page: int = Field(1, ge=1, description="Trang hiện tại")
    limit: int = Field(20, ge=1, le=100, description="Số videos mỗi trang")
    status: Optional[VideoStatus] = Field(None, description="Lọc theo status")
    video_type: Optional[VideoType] = Field(None, description="Lọc theo loại video")
    uploaded_by: Optional[str] = Field(None, description="Lọc theo người upload")
    search: Optional[str] = Field(None, description="Tìm kiếm theo title/description")
    sort_by: str = Field("uploaded_at", description="Sắp xếp theo field")
    sort_order: str = Field("desc", regex="^(asc|desc)$", description="Thứ tự sắp xếp")

    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "limit": 20,
                "status": "processed",
                "video_type": "cooking",
                "search": "phở",
                "sort_by": "view_count",
                "sort_order": "desc"
            }
        }

class VideoListResponse(BaseModel):
    """
    Response cho danh sách videos
    """
    videos: List[VideoMetadata] = Field(..., description="Danh sách videos")
    total: int = Field(..., description="Tổng số videos")
    page: int = Field(..., description="Trang hiện tại")
    limit: int = Field(..., description="Số videos mỗi trang")
    total_pages: int = Field(..., description="Tổng số trang")
    has_next: bool = Field(..., description="Có trang tiếp theo không")
    has_prev: bool = Field(..., description="Có trang trước không")

    class Config:
        schema_extra = {
            "example": {
                "videos": [
                    {
                        "video_id": "video_456",
                        "title": "Cách nấu phở bò",
                        "status": "processed",
                        "view_count": 245
                    }
                ],
                "total": 156,
                "page": 1,
                "limit": 20,
                "total_pages": 8,
                "has_next": True,
                "has_prev": False
            }
        }

class VideoManagementResponse(BaseModel):
    """
    Response cho admin video management
    """
    success: bool = Field(..., description="Thao tác có thành công không")
    total_videos: int = Field(..., description="Tổng số videos")
    page: int = Field(..., description="Trang hiện tại")
    limit: int = Field(..., description="Số videos mỗi trang")
    videos: List[VideoMetadata] = Field(..., description="Danh sách videos")
    filters_applied: Dict[str, Any] = Field(..., description="Filters đã áp dụng")
    admin_action_by: str = Field(..., description="Admin thực hiện action")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_videos": 1250,
                "page": 1,
                "limit": 50,
                "videos": [],
                "filters_applied": {
                    "status": "processing",
                    "video_type": "educational"
                },
                "admin_action_by": "xthanh1910"
            }
        }

# ================================
# STATISTICS & ANALYTICS MODELS
# ================================

class VideoStatistics(BaseModel):
    """
    Thống kê video system
    """
    total_videos: int = Field(..., description="Tổng số videos")
    videos_by_status: Dict[str, int] = Field(..., description="Videos theo status")
    videos_by_type: Dict[str, int] = Field(..., description="Videos theo loại")
    total_storage_gb: float = Field(..., description="Tổng dung lượng (GB)")
    avg_processing_time: float = Field(..., description="Thời gian xử lý trung bình")
    success_rate: float = Field(..., description="Tỷ lệ xử lý thành công")
    top_uploaders: List[Dict[str, Any]] = Field(..., description="Top người upload")

    class Config:
        schema_extra = {
            "example": {
                "total_videos": 1250,
                "videos_by_status": {
                    "processed": 1100,
                    "processing": 25,
                    "failed": 15,
                    "uploaded": 110
                },
                "videos_by_type": {
                    "cooking": 450,
                    "educational": 320,
                    "entertainment": 280,
                    "tutorial": 200
                },
                "total_storage_gb": 2450.7,
                "avg_processing_time": 180.5,
                "success_rate": 0.94,
                "top_uploaders": [
                    {
                        "user_id": "user_123",
                        "username": "chef_master",
                        "video_count": 89
                    }
                ]
            }
        }

class ProcessingStatistics(BaseModel):
    """
    Thống kê processing jobs
    """
    total_jobs: int = Field(..., description="Tổng số jobs")
    successful_jobs: int = Field(..., description="Jobs thành công")
    failed_jobs: int = Field(..., description="Jobs thất bại")
    running_jobs: int = Field(..., description="Jobs đang chạy")
    avg_time_minutes: float = Field(..., description="Thời gian trung bình (phút)")
    success_rate: float = Field(..., description="Tỷ lệ thành công")
    jobs_by_stage: Dict[str, int] = Field(..., description="Jobs theo stage")

    class Config:
        schema_extra = {
            "example": {
                "total_jobs": 1500,
                "successful_jobs": 1420,
                "failed_jobs": 65,
                "running_jobs": 15,
                "avg_time_minutes": 3.2,
                "success_rate": 0.947,
                "jobs_by_stage": {
                    "feature_extraction": 8,
                    "indexing": 4,
                    "speech_to_text": 3
                }
            }
        }

# ================================
# TEMPORARY & UTILITY MODELS
# ================================

class TempFileMetadata(BaseModel):
    """
    Metadata cho temporary files
    """
    temp_id: str = Field(..., description="ID của temp file")
    original_filename: str = Field(..., description="Tên file gốc")
    temp_path: str = Field(..., description="Đường dẫn temp file")
    file_size: int = Field(..., description="Kích thước file")
    uploaded_by: str = Field(..., description="Người upload")
    uploaded_at: datetime = Field(..., description="Thời gian upload")
    expires_at: datetime = Field(..., description="Thời gian hết hạn")
    download_count: int = Field(0, description="Số lần download")

    class Config:
        schema_extra = {
            "example": {
                "temp_id": "temp_123",
                "original_filename": "preview_video.mp4",
                "temp_path": "/uploads/temp/temp_123_preview_video.mp4",
                "file_size": 52428800,
                "uploaded_by": "user_456",
                "uploaded_at": "2025-07-03T14:16:45Z",
                "expires_at": "2025-07-04T14:16:45Z",
                "download_count": 3
            }
        }

class BulkActionRequest(BaseModel):
    """
    Request cho bulk actions trên videos
    """
    video_ids: List[str] = Field(..., min_items=1, max_items=100, description="Danh sách video IDs")
    action: str = Field(..., description="Action cần thực hiện")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters cho action")
    reason: Optional[str] = Field(None, description="Lý do thực hiện action")

    @validator('action')
    def validate_action(cls, v):
        allowed_actions = [
            'delete', 'archive', 'change_status', 'change_privacy',
            'reprocess', 'add_tags', 'remove_tags', 'change_type'
        ]
        if v not in allowed_actions:
            raise ValueError(f'Action không hợp lệ. Allowed: {allowed_actions}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "video_ids": ["video_123", "video_456", "video_789"],
                "action": "change_status",
                "parameters": {
                    "new_status": "archived"
                },
                "reason": "Bulk archive old videos"
            }
        }

class BulkActionResponse(BaseModel):
    """
    Response cho bulk actions
    """
    success: bool = Field(..., description="Bulk action có thành công không")
    total_videos: int = Field(..., description="Tổng số videos")
    successful_count: int = Field(..., description="Số videos xử lý thành công")
    failed_count: int = Field(..., description="Số videos xử lý thất bại")
    successful_videos: List[str] = Field(..., description="IDs videos thành công")
    failed_videos: List[Dict[str, str]] = Field(..., description="Videos thất bại với lý do")
    executed_by: str = Field(..., description="Người thực hiện")
    executed_at: datetime = Field(..., description="Thời gian thực hiện")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_videos": 3,
                "successful_count": 2,
                "failed_count": 1,
                "successful_videos": ["video_123", "video_456"],
                "failed_videos": [
                    {
                        "video_id": "video_789",
                        "error": "Video đang được xử lý, không thể archive"
                    }
                ],
                "executed_by": "xthanh1910",
                "executed_at": "2025-07-03T14:16:45Z"
            }
        }

# ================================
# UTILITY FUNCTIONS
# ================================

def create_video_upload_response(
    file_id: str,
    original_filename: str,
    file_size: int,
    status: VideoStatus = VideoStatus.UPLOADED,
    job_id: Optional[str] = None
) -> FileUploadResponse:
    """
    Helper function tạo video upload response
    """
    return FileUploadResponse(
        success=True,
        file_id=file_id,
        job_id=job_id,
        original_filename=original_filename,
        file_size=file_size,
        file_type=FileType.VIDEO,
        upload_path=f"/uploads/videos/{file_id}_{original_filename}",
        status=status,
        warnings=[],
        message="Video uploaded successfully" + (" and processing started" if job_id else "")
    )

def create_processing_job_info(
    job_id: str,
    video_id: str,
    status: JobStatus,
    progress: int = 0,
    current_stage: ProcessingStage = ProcessingStage.INITIALIZING
) -> ProcessingJobInfo:
    """
    Helper function tạo processing job info
    """
    return ProcessingJobInfo(
        job_id=job_id,
        video_id=video_id,
        status=status,
        progress=progress,
        current_stage=current_stage,
        started_at=datetime.utcnow() if status in [JobStatus.RUNNING, JobStatus.COMPLETED] else None,
        completed_at=datetime.utcnow() if status == JobStatus.COMPLETED else None,
        estimated_completion=None,
        error_message=None,
        result=None
    )

# ================================
# EXPORTS
# ================================

__all__ = [
    # Enums
    "VideoStatus",
    "VideoType",
    "ProcessingStage",
    "JobStatus",
    "FileType",

    # Upload Models
    "VideoUploadRequest",
    "FileUploadResponse",
    "BatchUploadResponse",

    # Processing Models
    "VideoProcessRequest",
    "ProcessingJobInfo",
    "ProcessingResult",

    # Metadata Models
    "VideoMetadata",
    "VideoSearchResult",

    # Management Models
    "VideoListRequest",
    "VideoListResponse",
    "VideoManagementResponse",

    # Statistics Models
    "VideoStatistics",
    "ProcessingStatistics",

    # Utility Models
    "TempFileMetadata",
    "BulkActionRequest",
    "BulkActionResponse",

    # Helper Functions
    "create_video_upload_response",
    "create_processing_job_info"
]