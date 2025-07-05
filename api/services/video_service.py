# api/services/video_service.py
"""
Video Service - Business logic cho video processing system
Dev2: API Integration & Services - kết nối API với Dev1's agents và Dev3's database
Current: 2025-07-03 14:04:03 UTC, User: xthanh1910
"""

import asyncio
import time
import os
import shutil
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from pathlib import Path

# Import agents từ Dev1 để xử lý video processing logic
from ..agents_manager import agents_manager
from agents.orchestrator.preprocessing_orchestrator import PreprocessingConfig

# Import database models/connections từ Dev3 (placeholder imports)
# Dev3 sẽ implement database thực tế cho video management
try:
    from database.models.video_models import VideoMetadata, ProcessingJob, VideoFile
    from database.models.user_models import UserActivity
    from database.connections.video_db import VideoDatabase
    from database.connections.user_db import UserDatabase
    from database.connections.job_db import JobDatabase
except ImportError:
    # Fallback nếu Dev3 chưa implement
    print("Warning: Video database models not found, using mock implementations")
    VideoMetadata = dict
    ProcessingJob = dict
    VideoFile = dict
    UserActivity = dict
    VideoDatabase = None
    UserDatabase = None
    JobDatabase = None

# Import cache service để lưu processing status
from .cache_service import CacheService

class VideoService:
    """
    Service xử lý logic video processing và management
    Dev2 chỉ làm integration - không viết AI logic phức tạp
    """

    def __init__(self):
        """
        Khởi tạo VideoService
        """
        self.cache_service = CacheService()

        # Database connections (Dev3's responsibility)
        self.video_db = VideoDatabase() if VideoDatabase else None
        self.user_db = UserDatabase() if UserDatabase else None
        self.job_db = JobDatabase() if JobDatabase else None

        # Mock storage nếu database chưa có
        self._mock_videos = {}      # video_id -> video_metadata
        self._mock_jobs = {}        # job_id -> job_data
        self._mock_files = {}       # file_id -> file_metadata

        # Service configuration
        self.max_concurrent_jobs = 5
        self.job_timeout_minutes = 60
        self.cleanup_temp_files_hours = 24
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

        # Directories configuration
        self.upload_dir = "uploads/videos"
        self.temp_dir = "uploads/temp"
        self.processed_dir = "uploads/processed"

        # Đảm bảo các thư mục tồn tại
        for directory in [self.upload_dir, self.temp_dir, self.processed_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"[{datetime.utcnow()}] VideoService initialized by user: xthanh1910")

    #======================================================================================================================================
    # MAIN VIDEO UPLOAD METHODS
    #======================================================================================================================================

    async def save_file_metadata(self, file_metadata: Dict[str, Any]) -> bool:
        """
        Lưu metadata của file đã upload vào database
        Dev2: Kết nối upload data → Dev3's database
        """
        try:
            # Validate required fields
            required_fields = ['file_id', 'original_filename', 'file_path', 'file_size', 'uploaded_by']
            for field in required_fields:
                if field not in file_metadata:
                    raise ValueError(f"Missing required field: {field}")

            # Thêm thông tin bổ sung
            file_metadata.update({
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'status': file_metadata.get('status', 'uploaded'),
                'processing_attempts': 0,
                'last_error': None
            })

            # Lưu vào database (Dev3)
            if self.video_db:
                success = await self.video_db.save_file_metadata(file_metadata)
            else:
                # Mock storage
                self._mock_files[file_metadata['file_id']] = file_metadata
                success = True

            # Lưu vào cache để access nhanh
            await self.cache_service.set(
                key=f"file_metadata:{file_metadata['file_id']}",
                value=file_metadata,
                ttl=3600  # 1 hour cache
            )

            print(f"[{datetime.utcnow()}] File metadata saved: {file_metadata['file_id']} by {file_metadata['uploaded_by']}")
            return success

        except Exception as e:
            print(f"Error saving file metadata: {str(e)}")
            return False

    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy metadata của file theo ID
        """
        try:
            # Thử lấy từ cache trước
            cached_data = await self.cache_service.get(f"file_metadata:{file_id}")
            if cached_data:
                return cached_data

            # Lấy từ database (Dev3)
            if self.video_db:
                metadata = await self.video_db.get_file_metadata(file_id)
            else:
                # Mock storage
                metadata = self._mock_files.get(file_id)

            # Lưu vào cache nếu tìm thấy
            if metadata:
                await self.cache_service.set(
                    key=f"file_metadata:{file_id}",
                    value=metadata,
                    ttl=3600
                )

            return metadata

        except Exception as e:
            print(f"Error getting file metadata: {str(e)}")
            return None

    #======================================================================================================================================
    # VIDEO PROCESSING METHODS
    #======================================================================================================================================

    async def start_video_processing(
        self,
        job_id: str,
        video_id: str,
        video_path: str,
        user_id: str,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Bắt đầu video processing job
        Dev2: Tạo job → gọi Dev1's preprocessing orchestrator → track progress
        """
        try:
            # Bước 1: Validate input
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Bước 2: Tạo job metadata
            job_data = {
                'job_id': job_id,
                'video_id': video_id,
                'video_path': video_path,
                'user_id': user_id,
                'status': 'pending',
                'created_at': datetime.utcnow(),
                'started_at': None,
                'completed_at': None,
                'progress': 0,
                'current_stage': 'initializing',
                'error_message': None,
                'config': config or {},
                'result': None
            }

            # Bước 3: Lưu job vào database (Dev3)
            await self._save_job_data(job_data)

            # Bước 4: Cập nhật file status
            await self._update_file_status(video_id, 'processing', job_id=job_id)

            # Bước 5: Bắt đầu processing trong background
            # Không await để không block API response
            asyncio.create_task(
                self._execute_video_processing_job(job_data)
            )

            print(f"[{datetime.utcnow()}] Video processing job started: {job_id} for video: {video_id} by user: {user_id}")
            return True

        except Exception as e:
            # Update job status thành failed nếu có lỗi ngay từ đầu
            await self._update_job_status(job_id, 'failed', error_message=str(e))
            print(f"Error starting video processing: {str(e)}")
            return False

    async def _execute_video_processing_job(self, job_data: Dict[str, Any]):
        """
        Thực thi video processing job (chạy trong background)
        """
        job_id = job_data['job_id']
        video_path = job_data['video_path']
        config = job_data.get('config', {})

        try:
            # Bước 1: Update job status thành 'running'
            await self._update_job_status(job_id, 'running', current_stage='starting', progress=5)

            # Bước 2: Tạo preprocessing config từ Dev1
            preprocessing_config = None
            if config:
                preprocessing_config = PreprocessingConfig(**config)

            # Bước 3: Lấy preprocessing orchestrator từ Dev1
            orchestrator = agents_manager.get_preprocessing_orchestrator()
            if not orchestrator:
                raise RuntimeError("Preprocessing orchestrator not available")

            # Bước 4: Update progress
            await self._update_job_status(job_id, 'running', current_stage='video_analysis', progress=15)

            # Bước 5: Gọi orchestrator để xử lý video (Dev1's AI logic)
            print(f"[{datetime.utcnow()}] Starting video processing for job: {job_id}")

            # Đây là nơi Dev1's agents thực sự làm việc
            result = orchestrator.process_video(
                video_path=video_path,
                config=preprocessing_config
            )

            # Bước 6: Kiểm tra kết quả từ agents
            if result.status == "error":
                await self._update_job_status(
                    job_id, 'failed',
                    error_message=result.error_message,
                    progress=0
                )
                return

            # Bước 7: Processing thành công
            pipeline_data = result.result

            # Update progress theo stages
            await self._update_job_status(job_id, 'running', current_stage='feature_extraction', progress=40)
            await asyncio.sleep(1)  # Simulate processing time

            await self._update_job_status(job_id, 'running', current_stage='knowledge_graph', progress=70)
            await asyncio.sleep(1)

            await self._update_job_status(job_id, 'running', current_stage='indexing', progress=90)
            await asyncio.sleep(1)

            # Bước 8: Hoàn thành job
            await self._update_job_status(
                job_id, 'completed',
                current_stage='finished',
                progress=100,
                result=pipeline_data
            )

            # Bước 9: Update file status thành 'processed'
            await self._update_file_status(
                job_data['video_id'],
                'processed',
                processing_result=pipeline_data
            )

            print(f"[{datetime.utcnow()}] Video processing completed successfully: {job_id}")

        except Exception as e:
            # Xử lý lỗi trong quá trình processing
            error_message = f"Processing failed: {str(e)}"
            await self._update_job_status(job_id, 'failed', error_message=error_message)
            await self._update_file_status(job_data['video_id'], 'failed', error_message=error_message)
            print(f"[{datetime.utcnow()}] Video processing failed: {job_id}, error: {error_message}")

    async def start_batch_processing(
        self,
        batch_job_id: str,
        batch_id: str,
        file_ids: List[str],
        user_id: str
    ) -> bool:
        """
        Bắt đầu batch processing cho nhiều videos
        """
        try:
            # Tạo batch job metadata
            batch_job_data = {
                'batch_job_id': batch_job_id,
                'batch_id': batch_id,
                'file_ids': file_ids,
                'user_id': user_id,
                'status': 'pending',
                'created_at': datetime.utcnow(),
                'total_files': len(file_ids),
                'completed_files': 0,
                'failed_files': 0,
                'individual_jobs': []
            }

            # Lưu batch job
            await self._save_batch_job_data(batch_job_data)

            # Tạo individual jobs cho từng file
            individual_jobs = []
            for i, file_id in enumerate(file_ids):
                # Lấy file metadata
                file_metadata = await self.get_file_metadata(file_id)
                if not file_metadata:
                    continue

                # Tạo individual job
                individual_job_id = f"{batch_job_id}_file_{i+1}"
                individual_jobs.append(individual_job_id)

                # Start processing cho file này
                await self.start_video_processing(
                    job_id=individual_job_id,
                    video_id=file_id,
                    video_path=file_metadata['file_path'],
                    user_id=user_id
                )

            # Update batch job với individual jobs
            batch_job_data['individual_jobs'] = individual_jobs
            await self._save_batch_job_data(batch_job_data)

            print(f"[{datetime.utcnow()}] Batch processing started: {batch_job_id} with {len(individual_jobs)} jobs")
            return True

        except Exception as e:
            print(f"Error starting batch processing: {str(e)}")
            return False

    #======================================================================================================================================
    # JOB STATUS & MANAGEMENT METHODS
    #======================================================================================================================================

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy status của processing job (để frontend hiển thị progress)
        """
        try:
            # Thử lấy từ cache trước (cho realtime updates)
            cached_status = await self.cache_service.get(f"job_status:{job_id}")
            if cached_status:
                return cached_status

            # Lấy từ database (Dev3)
            if self.job_db:
                job_data = await self.job_db.get_job_status(job_id)
            else:
                # Mock storage
                job_data = self._mock_jobs.get(job_id)

            if job_data:
                # Format status cho frontend Dev4
                status_data = {
                    'job_id': job_id,
                    'status': job_data.get('status', 'unknown'),
                    'progress': job_data.get('progress', 0),
                    'current_stage': job_data.get('current_stage', ''),
                    'started_at': job_data.get('started_at'),
                    'completed_at': job_data.get('completed_at'),
                    'error_message': job_data.get('error_message'),
                    'estimated_completion': self._estimate_completion_time(job_data),
                    'result': job_data.get('result')
                }

                # Cache status để access nhanh
                await self.cache_service.set(
                    key=f"job_status:{job_id}",
                    value=status_data,
                    ttl=60  # Cache 1 minute
                )

                return status_data

            return None

        except Exception as e:
            print(f"Error getting job status: {str(e)}")
            return None

    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """
        Hủy processing job đang chạy
        """
        try:
            # Lấy job data để check permissions
            job_data = await self._get_job_data(job_id)
            if not job_data:
                return False

            # Kiểm tra quyền (chỉ owner hoặc admin mới được hủy)
            if job_data.get('user_id') != user_id:
                # TODO: Check if user is admin
                return False

            # Chỉ có thể hủy job đang pending hoặc running
            current_status = job_data.get('status')
            if current_status not in ['pending', 'running']:
                return False

            # Update job status thành 'cancelled'
            await self._update_job_status(
                job_id, 'cancelled',
                error_message=f"Job cancelled by user: {user_id}"
            )

            # Update file status
            video_id = job_data.get('video_id')
            if video_id:
                await self._update_file_status(
                    video_id, 'cancelled',
                    error_message="Processing cancelled"
                )

            print(f"[{datetime.utcnow()}] Job cancelled: {job_id} by user: {user_id}")
            return True

        except Exception as e:
            print(f"Error cancelling job: {str(e)}")
            return False

    async def emergency_stop_all_processing(self) -> List[str]:
        """
        Dừng khẩn cấp tất cả processing jobs (admin only)
        """
        try:
            stopped_jobs = []

            if self.job_db:
                # Lấy tất cả jobs đang running từ database
                running_jobs = await self.job_db.get_running_jobs()
            else:
                # Mock storage
                running_jobs = [
                    job_data for job_data in self._mock_jobs.values()
                    if job_data.get('status') in ['pending', 'running']
                ]

            # Stop từng job
            for job in running_jobs:
                job_id = job.get('job_id')
                if job_id:
                    await self._update_job_status(
                        job_id, 'stopped',
                        error_message="Emergency stop by admin"
                    )
                    stopped_jobs.append(job_id)

            print(f"[{datetime.utcnow()}] Emergency stop executed: {len(stopped_jobs)} jobs stopped")
            return stopped_jobs

        except Exception as e:
            print(f"Error in emergency stop: {str(e)}")
            return []

    #======================================================================================================================================
    # VIDEO METADATA METHODS
    #======================================================================================================================================

    async def get_video_by_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin video theo ID
        """
        try:
            return await self.get_file_metadata(video_id)
        except Exception as e:
            print(f"Error getting video by ID: {str(e)}")
            return None

    async def get_video_detail_for_admin(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết video cho admin (bao gồm thông tin private)
        """
        try:
            # Lấy basic metadata
            video_metadata = await self.get_file_metadata(video_id)
            if not video_metadata:
                return None

            # Lấy thêm processing history
            processing_history = await self.get_video_processing_history(video_id)

            # Lấy user info của người upload
            uploader_id = video_metadata.get('uploaded_by')

            # Combine tất cả thông tin
            detailed_info = {
                **video_metadata,
                'processing_history': processing_history,
                'uploader_id': uploader_id,
                'admin_view': True,
                'retrieved_at': datetime.utcnow()
            }

            return detailed_info

        except Exception as e:
            print(f"Error getting video detail for admin: {str(e)}")
            return None

    async def get_videos_list(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lấy danh sách videos với filters và pagination
        """
        try:
            page = filters.get('page', 1)
            limit = filters.get('limit', 20)
            status_filter = filters.get('status')
            user_filter = filters.get('uploaded_by')

            if self.video_db:
                # Lấy từ database (Dev3)
                result = await self.video_db.get_videos_list(filters)
            else:
                # Mock storage
                all_videos = list(self._mock_files.values())

                # Apply filters
                if status_filter:
                    all_videos = [v for v in all_videos if v.get('status') == status_filter]
                if user_filter:
                    all_videos = [v for v in all_videos if v.get('uploaded_by') == user_filter]

                # Pagination
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                paginated_videos = all_videos[start_idx:end_idx]

                result = {
                    'videos': paginated_videos,
                    'total': len(all_videos),
                    'page': page,
                    'limit': limit,
                    'total_pages': (len(all_videos) + limit - 1) // limit
                }

            return result

        except Exception as e:
            print(f"Error getting videos list: {str(e)}")
            return {'videos': [], 'total': 0, 'page': page, 'limit': limit}

    async def get_videos_for_admin(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lấy danh sách videos cho admin (bao gồm tất cả videos)
        """
        try:
            # Admin có thể xem tất cả videos
            filters['admin_view'] = True
            return await self.get_videos_list(filters)

        except Exception as e:
            print(f"Error getting videos for admin: {str(e)}")
            return {'videos': [], 'total': 0}

    async def get_videos_by_user(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Lấy videos của một user cụ thể
        """
        try:
            filters = {
                'uploaded_by': user_id,
                'limit': limit,
                'page': 1
            }

            result = await self.get_videos_list(filters)
            return result.get('videos', [])

        except Exception as e:
            print(f"Error getting videos by user: {str(e)}")
            return []

    #======================================================================================================================================
    # PROCESSING HISTORY & ANALYTICS
    #======================================================================================================================================

    async def get_video_processing_history(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử processing của video
        """
        try:
            if self.job_db:
                history = await self.job_db.get_processing_history_by_video(video_id)
            else:
                # Mock storage
                history = []
                for job_data in self._mock_jobs.values():
                    if job_data.get('video_id') == video_id:
                        history.append({
                            'job_id': job_data.get('job_id'),
                            'status': job_data.get('status'),
                            'started_at': job_data.get('started_at'),
                            'completed_at': job_data.get('completed_at'),
                            'error_message': job_data.get('error_message'),
                            'progress': job_data.get('progress', 0)
                        })

                # Sort by creation time
                history.sort(key=lambda x: x.get('started_at') or '0', reverse=True)

            return history

        except Exception as e:
            print(f"Error getting processing history: {str(e)}")
            return []

    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê processing jobs
        """
        try:
            if self.job_db:
                stats = await self.job_db.get_processing_statistics()
            else:
                # Mock statistics
                total_jobs = len(self._mock_jobs)
                successful_jobs = len([j for j in self._mock_jobs.values() if j.get('status') == 'completed'])
                failed_jobs = len([j for j in self._mock_jobs.values() if j.get('status') == 'failed'])
                running_jobs = len([j for j in self._mock_jobs.values() if j.get('status') == 'running'])

                stats = {
                    'total_jobs': total_jobs,
                    'successful_jobs': successful_jobs,
                    'failed_jobs': failed_jobs,
                    'running_jobs': running_jobs,
                    'success_rate': successful_jobs / max(total_jobs, 1) * 100,
                    'avg_time_minutes': 15.5  # Mock average time
                }

            return stats

        except Exception as e:
            print(f"Error getting processing stats: {str(e)}")
            return {}

    async def get_admin_video_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê videos cho admin dashboard
        """
        try:
            if self.video_db:
                stats = await self.video_db.get_admin_video_statistics()
            else:
                # Mock statistics
                total_videos = len(self._mock_files)
                processing_count = len([f for f in self._mock_files.values() if f.get('status') == 'processing'])
                completed_count = len([f for f in self._mock_files.values() if f.get('status') == 'processed'])
                failed_count = len([f for f in self._mock_files.values() if f.get('status') == 'failed'])

                # Mock storage calculations
                total_storage_gb = sum(f.get('file_size', 0) for f in self._mock_files.values()) / (1024**3)

                stats = {
                    'total_videos': total_videos,
                    'processing_count': processing_count,
                    'completed_count': completed_count,
                    'failed_count': failed_count,
                    'total_storage_gb': round(total_storage_gb, 2),
                    'available_storage_gb': 1000.0  # Mock available storage
                }

            return stats

        except Exception as e:
            print(f"Error getting admin video stats: {str(e)}")
            return {}

    #======================================================================================================================================
    # FILE MANAGEMENT METHODS
    #======================================================================================================================================

    async def delete_file_metadata(self, file_id: str) -> bool:
        """
        Xóa file metadata khỏi database
        """
        try:
            if self.video_db:
                success = await self.video_db.delete_file_metadata(file_id)
            else:
                # Mock storage
                if file_id in self._mock_files:
                    del self._mock_files[file_id]
                success = True

            # Xóa khỏi cache
            await self.cache_service.delete(f"file_metadata:{file_id}")

            return success

        except Exception as e:
            print(f"Error deleting file metadata: {str(e)}")
            return False

    async def delete_video_completely(
        self,
        video_id: str,
        deleted_by: str,
        reason: str = ""
    ) -> bool:
        """
        Xóa video hoàn toàn (file + metadata + processing data)
        """
        try:
            # Lấy file metadata để biết file path
            file_metadata = await self.get_file_metadata(video_id)
            if not file_metadata:
                return False

            file_path = file_metadata.get('file_path')

            # Xóa file khỏi disk
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"[{datetime.utcnow()}] Deleted file: {file_path}")

            # Xóa metadata khỏi database
            await self.delete_file_metadata(video_id)

            # Xóa processing jobs liên quan
            await self._delete_jobs_by_video_id(video_id)

            # Log deletion activity
            await self._log_video_deletion(video_id, deleted_by, reason, file_metadata)

            print(f"[{datetime.utcnow()}] Video deleted completely: {video_id} by {deleted_by}")
            return True

        except Exception as e:
            print(f"Error deleting video completely: {str(e)}")
            return False

    async def get_user_uploads(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lấy danh sách uploads của user
        """
        try:
            return await self.get_videos_list(filters)
        except Exception as e:
            print(f"Error getting user uploads: {str(e)}")
            return {'videos': [], 'total': 0}

    #======================================================================================================================================
    # TEMP FILE METHODS
    #======================================================================================================================================

    async def save_temp_file_metadata(self, temp_metadata: Dict[str, Any]) -> bool:
        """
        Lưu thông tin temp file vào cache
        """
        try:
            temp_id = temp_metadata['temp_id']

            # Lưu vào cache với TTL
            expires_at = temp_metadata.get('expires_at', time.time() + 86400)  # Default 24h
            ttl = max(int(expires_at - time.time()), 60)  # Minimum 1 minute

            await self.cache_service.set(
                key=f"temp_file:{temp_id}",
                value=temp_metadata,
                ttl=ttl
            )

            return True

        except Exception as e:
            print(f"Error saving temp file metadata: {str(e)}")
            return False

    async def get_temp_file_metadata(self, temp_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin temp file
        """
        try:
            return await self.cache_service.get(f"temp_file:{temp_id}")
        except Exception as e:
            print(f"Error getting temp file metadata: {str(e)}")
            return None

    #======================================================================================================================================
    # HELPER METHODS
    #======================================================================================================================================

    async def _save_job_data(self, job_data: Dict[str, Any]):
        """
        Lưu job data vào database
        """
        try:
            if self.job_db:
                await self.job_db.save_job_data(job_data)
            else:
                # Mock storage
                self._mock_jobs[job_data['job_id']] = job_data

            # Cache job status để access nhanh
            await self.cache_service.set(
                key=f"job_status:{job_data['job_id']}",
                value=job_data,
                ttl=300  # 5 minutes
            )

        except Exception as e:
            print(f"Error saving job data: {str(e)}")

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        current_stage: str = None,
        progress: int = None,
        error_message: str = None,
        result: Dict = None
    ):
        """
        Cập nhật status của job
        """
        try:
            # Lấy job data hiện tại
            job_data = await self._get_job_data(job_id)
            if not job_data:
                return

            # Update fields
            job_data['status'] = status
            job_data['updated_at'] = datetime.utcnow()

            if current_stage:
                job_data['current_stage'] = current_stage
            if progress is not None:
                job_data['progress'] = progress
            if error_message:
                job_data['error_message'] = error_message
            if result:
                job_data['result'] = result

            # Set completed_at nếu job finished
            if status in ['completed', 'failed', 'cancelled', 'stopped']:
                job_data['completed_at'] = datetime.utcnow()
            elif status == 'running' and not job_data.get('started_at'):
                job_data['started_at'] = datetime.utcnow()

            # Lưu vào database và cache
            await self._save_job_data(job_data)

        except Exception as e:
            print(f"Error updating job status: {str(e)}")

    async def _get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy job data từ database hoặc cache
        """
        try:
            # Thử cache trước
            cached_data = await self.cache_service.get(f"job_status:{job_id}")
            if cached_data:
                return cached_data

            # Lấy từ database
            if self.job_db:
                return await self.job_db.get_job_data(job_id)
            else:
                return self._mock_jobs.get(job_id)

        except Exception as e:
            print(f"Error getting job data: {str(e)}")
            return None

    async def _update_file_status(
        self,
        file_id: str,
        status: str,
        job_id: str = None,
        error_message: str = None,
        processing_result: Dict = None
    ):
        """
        Cập nhật status của file
        """
        try:
            # Lấy file metadata hiện tại
            file_metadata = await self.get_file_metadata(file_id)
            if not file_metadata:
                return

            # Update status
            file_metadata['status'] = status
            file_metadata['updated_at'] = datetime.utcnow()

            if job_id:
                file_metadata['current_job_id'] = job_id
            if error_message:
                file_metadata['last_error'] = error_message
            if processing_result:
                file_metadata['processing_result'] = processing_result

            # Lưu lại
            await self.save_file_metadata(file_metadata)

        except Exception as e:
            print(f"Error updating file status: {str(e)}")

    def _estimate_completion_time(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Ước tính thời gian hoàn thành job
        """
        try:
            status = job_data.get('status')
            if status in ['completed', 'failed', 'cancelled']:
                return None

            started_at = job_data.get('started_at')
            if not started_at:
                return "Estimating..."

            progress = job_data.get('progress', 0)
            if progress <= 0:
                return "Estimating..."

            # Tính time elapsed và estimated total time
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))

            elapsed_seconds = (datetime.utcnow() - started_at).total_seconds()
            estimated_total_seconds = elapsed_seconds * (100 / progress)
            remaining_seconds = estimated_total_seconds - elapsed_seconds

            if remaining_seconds <= 0:
                return "Finishing up..."

            # Convert to human readable
            if remaining_seconds < 60:
                return f"~{int(remaining_seconds)} seconds"
            elif remaining_seconds < 3600:
                return f"~{int(remaining_seconds/60)} minutes"
            else:
                return f"~{int(remaining_seconds/3600)} hours"

        except Exception as e:
            return "Unknown"

    async def _save_batch_job_data(self, batch_job_data: Dict[str, Any]):
        """
        Lưu batch job data
        """
        try:
            if self.job_db:
                await self.job_db.save_batch_job_data(batch_job_data)
            else:
                # Mock storage - add to regular jobs with batch prefix
                self._mock_jobs[f"batch_{batch_job_data['batch_job_id']}"] = batch_job_data

        except Exception as e:
            print(f"Error saving batch job data: {str(e)}")

    async def _delete_jobs_by_video_id(self, video_id: str):
        """
        Xóa tất cả jobs liên quan đến video
        """
        try:
            if self.job_db:
                await self.job_db.delete_jobs_by_video_id(video_id)
            else:
                # Mock storage
                jobs_to_delete = [
                    job_id for job_id, job_data in self._mock_jobs.items()
                    if job_data.get('video_id') == video_id
                ]
                for job_id in jobs_to_delete:
                    del self._mock_jobs[job_id]

        except Exception as e:
            print(f"Error deleting jobs by video ID: {str(e)}")

    async def _log_video_deletion(
        self,
        video_id: str,
        deleted_by: str,
        reason: str,
        file_metadata: Dict[str, Any]
    ):
        """
        Log video deletion activity
        """
        try:
            deletion_log = {
                'video_id': video_id,
                'deleted_by': deleted_by,
                'reason': reason,
                'original_filename': file_metadata.get('original_filename'),
                'file_size': file_metadata.get('file_size'),
                'uploaded_by': file_metadata.get('uploaded_by'),
                'deleted_at': datetime.utcnow(),
                'action_type': 'video_deletion'
            }

            if self.video_db:
                await self.video_db.log_deletion_activity(deletion_log)
            else:
                print(f"Video deletion logged: {deletion_log}")

        except Exception as e:
            print(f"Error logging video deletion: {str(e)}")

    #======================================================================================================================================
    # SERVICE MANAGEMENT METHODS
    #======================================================================================================================================

    async def get_total_video_count(self) -> int:
        """
        Lấy tổng số videos trong hệ thống
        """
        try:
            if self.video_db:
                return await self.video_db.get_total_video_count()
            else:
                return len(self._mock_files)
        except Exception as e:
            print(f"Error getting total video count: {str(e)}")
            return 0

    async def get_processing_queue_size(self) -> int:
        """
        Lấy số job đang trong queue
        """
        try:
            if self.job_db:
                return await self.job_db.get_processing_queue_size()
            else:
                return len([j for j in self._mock_jobs.values() if j.get('status') in ['pending', 'running']])
        except Exception as e:
            print(f"Error getting processing queue size: {str(e)}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check cho VideoService
        """
        try:
            start_time = time.time()

            # Test preprocessing orchestrator connection (Dev1)
            orchestrator = agents_manager.get_preprocessing_orchestrator()
            orchestrator_ok = orchestrator is not None

            # Test database connections (Dev3)
            db_ok = True
            if self.video_db:
                try:
                    db_ok = await self.video_db.health_check()
                except Exception:
                    db_ok = False

            # Test cache connection
            cache_result = await self.cache_service.health_check()
            cache_ok = cache_result.get("status") == "healthy"

            # Test file system
            upload_dir_ok = os.path.exists(self.upload_dir) and os.path.isdir(self.upload_dir)

            # Get disk space
            disk_usage = shutil.disk_usage(self.upload_dir)
            free_space_gb = disk_usage.free / (1024**3)

            response_time = (time.time() - start_time) * 1000
            overall_status = "healthy" if (orchestrator_ok and db_ok and cache_ok and upload_dir_ok) else "degraded"

            return {
                "status": overall_status,
                "service": "video_service",
                "response_time_ms": round(response_time, 2),
                "components": {
                    "preprocessing_orchestrator": "healthy" if orchestrator_ok else "error",
                    "database": "healthy" if db_ok else "error",
                    "cache": "healthy" if cache_ok else "error",
                    "file_system": "healthy" if upload_dir_ok else "error"
                },
                "metrics": {
                    "total_videos": len(self._mock_files),
                    "active_jobs": len([j for j in self._mock_jobs.values() if j.get('status') in ['pending', 'running']]),
                    "free_space_gb": round(free_space_gb, 2),
                    "supported_formats": self.supported_video_formats,
                    "max_concurrent_jobs": self.max_concurrent_jobs
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def initialize(self):
        """
        Khởi tạo VideoService khi app startup
        """
        try:
            # Initialize database connections (Dev3)
            if self.video_db:
                await self.video_db.connect()
            if self.job_db:
                await self.job_db.connect()
            if self.user_db:
                await self.user_db.connect()

            # Initialize cache service
            await self.cache_service.initialize()

            # Ensure directories exist
            for directory in [self.upload_dir, self.temp_dir, self.processed_dir]:
                os.makedirs(directory, exist_ok=True)

            print(f"[{datetime.utcnow()}] VideoService initialized successfully")

        except Exception as e:
            print(f"[{datetime.utcnow()}] VideoService initialization failed: {str(e)}")
            raise

    async def shutdown(self):
        """
        Graceful shutdown VideoService
        """
        try:
            # Stop all running jobs gracefully
            await self.emergency_stop_all_processing()

            # Close database connections (Dev3)
            if self.video_db:
                await self.video_db.disconnect()
            if self.job_db:
                await self.job_db.disconnect()
            if self.user_db:
                await self.user_db.disconnect()

            # Close cache service
            await self.cache_service.close()

            print(f"[{datetime.utcnow()}] VideoService shutdown completed")

        except Exception as e:
            print(f"[{datetime.utcnow()}] VideoService shutdown error: {str(e)}")

# Export service instance
video_service = VideoService()