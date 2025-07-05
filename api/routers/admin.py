# api/routers/admin.py
"""
Admin Router - API endpoints cho admin management và monitoring
Dev2: API Integration & Routing - tích hợp admin features từ các services
Current: 2025-07-03 12:03:13 UTC, User: xthanh1910 (Admin)
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
from datetime import datetime, timedelta

# Import models cho admin API (Dev2 tạo đơn giản)
from ..models.user_models import AdminUser, User
from ..models.video_models import VideoStatus
from ..models.admin_models import (
    SystemStats, UserManagementResponse, VideoManagementResponse,
    AdminDashboard, SystemHealth
)

# Import services để connect với database Dev3 và agents Dev1
from ..services.user_service import UserService
from ..services.video_service import VideoService
from ..services.admin_service import AdminService
from ..middleware.auth import get_current_admin_user, get_current_user

# Import agents manager để lấy system health từ Dev1
from ..agents_manager import agents_manager

# Khởi tạo services
user_service = UserService()
video_service = VideoService()
admin_service = AdminService()
router = APIRouter(prefix="/admin", tags=["admin"])

#==========================================================================================================================================
# ADMIN DASHBOARD ENDPOINTS
#==========================================================================================================================================

@router.get("/dashboard", response_model=AdminDashboard)
async def get_admin_dashboard(
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Dashboard tổng quan cho admin
    Hiển thị tất cả thống kê quan trọng của hệ thống
    """
    try:
        # Bước 1: Lấy thống kê users từ Dev3's database
        user_stats = await user_service.get_admin_user_stats()

        # Bước 2: Lấy thống kê videos từ Dev3's database
        video_stats = await video_service.get_admin_video_stats()

        # Bước 3: Lấy system health từ Dev1's agents
        agents_health = agents_manager.get_system_health_summary()

        # Bước 4: Lấy processing stats
        processing_stats = await video_service.get_processing_stats()

        # Bước 5: Lấy recent activities
        recent_activities = await admin_service.get_recent_activities(limit=20)

        # Bước 6: Tính toán uptime
        uptime_hours = agents_health.get("uptime_hours", 0)

        # Bước 7: Tạo dashboard response
        dashboard = AdminDashboard(
            # System overview
            system_status=agents_health.get("overall_status", "unknown"),
            uptime_hours=uptime_hours,
            current_admin=admin_user.username,
            dashboard_generated_at=datetime.utcnow(),

            # User statistics
            total_users=user_stats.get("total_users", 0),
            active_users_today=user_stats.get("active_today", 0),
            new_users_this_week=user_stats.get("new_this_week", 0),

            # Video statistics
            total_videos=video_stats.get("total_videos", 0),
            videos_processing=video_stats.get("processing_count", 0),
            videos_completed=video_stats.get("completed_count", 0),
            videos_failed=video_stats.get("failed_count", 0),

            # Processing statistics
            total_processing_jobs=processing_stats.get("total_jobs", 0),
            successful_jobs=processing_stats.get("successful_jobs", 0),
            failed_jobs=processing_stats.get("failed_jobs", 0),
            avg_processing_time_minutes=processing_stats.get("avg_time_minutes", 0),

            # Storage statistics
            total_storage_gb=video_stats.get("total_storage_gb", 0),
            available_storage_gb=video_stats.get("available_storage_gb", 0),

            # Agent health
            total_agents=agents_health.get("total_agents", 0),
            healthy_agents=agents_health.get("healthy_agents", 0),
            degraded_agents=agents_health.get("degraded_agents", 0),

            # Recent activities
            recent_activities=recent_activities
        )

        return dashboard

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard load failed: {str(e)}")

@router.get("/stats/summary")
async def get_system_summary(
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Tóm tắt nhanh system stats cho admin
    """
    try:
        # Lấy stats cơ bản từ các services
        user_count = await user_service.get_total_user_count()
        video_count = await video_service.get_total_video_count()
        processing_queue = await video_service.get_processing_queue_size()
        system_health = agents_manager.get_system_health_summary()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "admin": admin_user.username,
            "summary": {
                "users": {
                    "total": user_count,
                    "status": "normal" if user_count > 0 else "no_users"
                },
                "videos": {
                    "total": video_count,
                    "processing_queue": processing_queue,
                    "status": "normal" if processing_queue < 100 else "high_load"
                },
                "system": {
                    "health": system_health.get("overall_status", "unknown"),
                    "agents": f"{system_health.get('healthy_agents', 0)}/{system_health.get('total_agents', 0)}",
                    "uptime_hours": round(system_health.get("uptime_hours", 0), 1)
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")

#==========================================================================================================================================
# USER MANAGEMENT ENDPOINTS
#==========================================================================================================================================

@router.get("/users", response_model=UserManagementResponse)
async def get_users_list(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),  # active, inactive, banned
    search: Optional[str] = Query(None),  # Tìm theo username/email
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy danh sách users với phân trang và filter
    """
    try:
        # Tạo filter parameters cho Dev3's database
        filters = {
            "page": page,
            "limit": limit
        }

        if status:
            filters["status"] = status
        if search:
            filters["search"] = search

        # Lấy danh sách users từ Dev3's service
        users_data = await user_service.get_users_for_admin(filters)

        return UserManagementResponse(
            success=True,
            total_users=users_data.get("total", 0),
            page=page,
            limit=limit,
            users=users_data.get("users", []),
            filters_applied=filters,
            admin_action_by=admin_user.username
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get users failed: {str(e)}")

@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: str,
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy thông tin chi tiết của một user
    """
    try:
        # Lấy user info từ Dev3's database
        user_detail = await user_service.get_user_detail_for_admin(user_id)

        if not user_detail:
            raise HTTPException(status_code=404, detail="User không tồn tại")

        # Lấy thêm user activities
        user_activities = await user_service.get_user_activities(user_id, limit=50)

        # Lấy videos của user
        user_videos = await video_service.get_videos_by_user(user_id, limit=20)

        return {
            "user_info": user_detail,
            "recent_activities": user_activities,
            "uploaded_videos": user_videos,
            "admin_viewed_by": admin_user.username,
            "viewed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get user detail failed: {str(e)}")

@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    new_status: str,  # active, inactive, banned
    reason: str = "",
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Cập nhật trạng thái user (ban, unban, deactivate)
    """
    try:
        # Validate status
        valid_statuses = ["active", "inactive", "banned"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )

        # Kiểm tra user có tồn tại không
        user_detail = await user_service.get_user_detail_for_admin(user_id)
        if not user_detail:
            raise HTTPException(status_code=404, detail="User không tồn tại")

        # Không cho phép admin tự ban chính mình
        if user_id == admin_user.user_id and new_status == "banned":
            raise HTTPException(status_code=400, detail="Không thể ban chính mình")

        # Cập nhật status qua Dev3's service
        success = await user_service.update_user_status(
            user_id=user_id,
            new_status=new_status,
            reason=reason,
            updated_by=admin_user.user_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Cập nhật status thất bại")

        # Log admin action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="update_user_status",
            target_id=user_id,
            details={
                "old_status": user_detail.get("status"),
                "new_status": new_status,
                "reason": reason
            }
        )

        return {
            "success": True,
            "user_id": user_id,
            "old_status": user_detail.get("status"),
            "new_status": new_status,
            "reason": reason,
            "updated_by": admin_user.username,
            "updated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update user status failed: {str(e)}")

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    confirm: bool = Query(False),  # Bắt buộc confirm để tránh xóa nhầm
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Xóa user khỏi hệ thống (nguy hiểm - cần confirm)
    """
    try:
        # Bắt buộc phải confirm
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion by setting confirm=true"
            )

        # Kiểm tra user có tồn tại không
        user_detail = await user_service.get_user_detail_for_admin(user_id)
        if not user_detail:
            raise HTTPException(status_code=404, detail="User không tồn tại")

        # Không cho phép admin xóa chính mình
        if user_id == admin_user.user_id:
            raise HTTPException(status_code=400, detail="Không thể xóa chính mình")

        # Xóa user và tất cả dữ liệu liên quan qua Dev3's service
        # Bao gồm: user profile, videos, chat history, etc.
        success = await user_service.delete_user_completely(
            user_id=user_id,
            deleted_by=admin_user.user_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Xóa user thất bại")

        # Log admin action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="delete_user",
            target_id=user_id,
            details={
                "user_info": user_detail,
                "deletion_confirmed": True
            }
        )

        return {
            "success": True,
            "user_id": user_id,
            "username": user_detail.get("username"),
            "deleted_by": admin_user.username,
            "deleted_at": datetime.utcnow().isoformat(),
            "message": "User và tất cả dữ liệu liên quan đã được xóa"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete user failed: {str(e)}")

#==========================================================================================================================================
# VIDEO MANAGEMENT ENDPOINTS
#==========================================================================================================================================

@router.get("/videos", response_model=VideoManagementResponse)
async def get_videos_list(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[VideoStatus] = Query(None),
    user_id: Optional[str] = Query(None),  # Lọc theo user
    search: Optional[str] = Query(None),   # Tìm theo tên file
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy danh sách videos với quyền admin (xem tất cả)
    """
    try:
        # Tạo filter parameters
        filters = {
            "page": page,
            "limit": limit,
            "admin_view": True  # Admin có thể xem tất cả videos
        }

        if status:
            filters["status"] = status.value
        if user_id:
            filters["uploaded_by"] = user_id
        if search:
            filters["search"] = search

        # Lấy danh sách videos từ Dev3's service
        videos_data = await video_service.get_videos_for_admin(filters)

        return VideoManagementResponse(
            success=True,
            total_videos=videos_data.get("total", 0),
            page=page,
            limit=limit,
            videos=videos_data.get("videos", []),
            filters_applied=filters,
            admin_action_by=admin_user.username
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get videos failed: {str(e)}")

@router.get("/videos/{video_id}")
async def get_video_detail(
    video_id: str,
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy thông tin chi tiết video với quyền admin
    """
    try:
        # Lấy video detail từ Dev3's database
        video_detail = await video_service.get_video_detail_for_admin(video_id)

        if not video_detail:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Lấy processing history nếu có
        processing_history = await video_service.get_video_processing_history(video_id)

        # Lấy user info của người upload
        uploader_info = await user_service.get_user_basic_info(
            video_detail.get("uploaded_by")
        )

        return {
            "video_info": video_detail,
            "processing_history": processing_history,
            "uploader_info": uploader_info,
            "admin_viewed_by": admin_user.username,
            "viewed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get video detail failed: {str(e)}")

@router.put("/videos/{video_id}/status")
async def update_video_status(
    video_id: str,
    new_status: VideoStatus,
    reason: str = "",
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Cập nhật status video (approve, reject, remove)
    """
    try:
        # Kiểm tra video có tồn tại không
        video_detail = await video_service.get_video_detail_for_admin(video_id)
        if not video_detail:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Cập nhật status qua Dev3's service
        success = await video_service.update_video_status_by_admin(
            video_id=video_id,
            new_status=new_status.value,
            reason=reason,
            updated_by=admin_user.user_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Cập nhật video status thất bại")

        # Log admin action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="update_video_status",
            target_id=video_id,
            details={
                "old_status": video_detail.get("status"),
                "new_status": new_status.value,
                "reason": reason
            }
        )

        return {
            "success": True,
            "video_id": video_id,
            "old_status": video_detail.get("status"),
            "new_status": new_status.value,
            "reason": reason,
            "updated_by": admin_user.username,
            "updated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update video status failed: {str(e)}")

@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    confirm: bool = Query(False),
    reason: str = "",
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Xóa video khỏi hệ thống (bao gồm file và metadata)
    """
    try:
        # Bắt buộc confirm
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion by setting confirm=true"
            )

        # Kiểm tra video có tồn tại không
        video_detail = await video_service.get_video_detail_for_admin(video_id)
        if not video_detail:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Xóa video (file + metadata + processing data) qua Dev3's service
        success = await video_service.delete_video_completely(
            video_id=video_id,
            deleted_by=admin_user.user_id,
            reason=reason
        )

        if not success:
            raise HTTPException(status_code=500, detail="Xóa video thất bại")

        # Log admin action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="delete_video",
            target_id=video_id,
            details={
                "video_info": video_detail,
                "reason": reason,
                "deletion_confirmed": True
            }
        )

        return {
            "success": True,
            "video_id": video_id,
            "filename": video_detail.get("original_filename"),
            "reason": reason,
            "deleted_by": admin_user.username,
            "deleted_at": datetime.utcnow().isoformat(),
            "message": "Video và tất cả dữ liệu liên quan đã được xóa"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete video failed: {str(e)}")

#==========================================================================================================================================
# SYSTEM MONITORING ENDPOINTS
#==========================================================================================================================================

@router.get("/system/health", response_model=SystemHealth)
async def get_system_health(
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Kiểm tra tình trạng sức khỏe toàn bộ hệ thống
    """
    try:
        # Lấy health từ Dev1's agents
        agents_health = agents_manager.get_system_health_summary()

        # Kiểm tra database health từ Dev3's services
        db_health = await admin_service.check_database_health()

        # Kiểm tra storage health
        storage_health = await admin_service.check_storage_health()

        # Kiểm tra external services health
        external_health = await admin_service.check_external_services_health()

        # Tính overall health
        all_components_healthy = (
            agents_health.get("overall_status") == "healthy" and
            db_health.get("status") == "healthy" and
            storage_health.get("status") == "healthy" and
            external_health.get("status") == "healthy"
        )

        overall_status = "healthy" if all_components_healthy else "degraded"

        return SystemHealth(
            overall_status=overall_status,
            check_time=datetime.utcnow(),
            checked_by=admin_user.username,
            agents_health=agents_health,
            database_health=db_health,
            storage_health=storage_health,
            external_services_health=external_health,
            recommendations=_get_health_recommendations(
                agents_health, db_health, storage_health, external_health
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@router.get("/system/performance")
async def get_system_performance(
    hours: int = Query(24, ge=1, le=168),  # Tối đa 1 tuần
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy thống kê performance trong X giờ qua
    """
    try:
        # Lấy performance metrics từ Dev3's database
        performance_data = await admin_service.get_performance_metrics(hours=hours)

        return {
            "time_range_hours": hours,
            "generated_by": admin_user.username,
            "generated_at": datetime.utcnow().isoformat(),
            "performance_metrics": performance_data,
            "summary": {
                "avg_response_time_ms": performance_data.get("avg_response_time", 0),
                "total_requests": performance_data.get("total_requests", 0),
                "error_rate_percent": performance_data.get("error_rate", 0),
                "peak_concurrent_users": performance_data.get("peak_users", 0)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")

#==========================================================================================================================================
# ADMIN ACTIONS LOG
#==========================================================================================================================================

@router.get("/actions/log")
async def get_admin_actions_log(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    admin_id: Optional[str] = Query(None),  # Filter theo admin
    action_type: Optional[str] = Query(None),  # Filter theo loại action
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Lấy log các actions của admin (audit trail)
    """
    try:
        # Tạo filter parameters
        filters = {
            "page": page,
            "limit": limit
        }

        if admin_id:
            filters["admin_id"] = admin_id
        if action_type:
            filters["action_type"] = action_type

        # Lấy log từ Dev3's database
        actions_log = await admin_service.get_admin_actions_log(filters)

        return {
            "total_actions": actions_log.get("total", 0),
            "page": page,
            "limit": limit,
            "actions": actions_log.get("actions", []),
            "filters": filters,
            "viewed_by": admin_user.username,
            "viewed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get actions log failed: {str(e)}")

#==========================================================================================================================================
# EMERGENCY CONTROLS
#==========================================================================================================================================

@router.post("/emergency/stop-processing")
async def emergency_stop_processing(
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Dừng khẩn cấp tất cả video processing jobs
    """
    try:
        # Dừng tất cả processing jobs qua Dev3's service
        stopped_jobs = await video_service.emergency_stop_all_processing()

        # Log emergency action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="emergency_stop_processing",
            target_id="all_jobs",
            details={
                "stopped_jobs_count": len(stopped_jobs),
                "stopped_jobs": stopped_jobs
            }
        )

        return {
            "success": True,
            "action": "emergency_stop_processing",
            "stopped_jobs_count": len(stopped_jobs),
            "stopped_jobs": stopped_jobs,
            "executed_by": admin_user.username,
            "executed_at": datetime.utcnow().isoformat(),
            "message": "Tất cả processing jobs đã được dừng khẩn cấp"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

@router.post("/emergency/clear-cache")
async def emergency_clear_cache(
    admin_user: AdminUser = Depends(get_current_admin_user)
):
    """
    Xóa cache khẩn cấp để giải phóng memory
    """
    try:
        # Clear cache qua services
        cache_cleared = await admin_service.emergency_clear_cache()

        # Log emergency action
        await admin_service.log_admin_action(
            admin_id=admin_user.user_id,
            action_type="emergency_clear_cache",
            target_id="system_cache",
            details=cache_cleared
        )

        return {
            "success": True,
            "action": "emergency_clear_cache",
            "cache_cleared": cache_cleared,
            "executed_by": admin_user.username,
            "executed_at": datetime.utcnow().isoformat(),
            "message": "Cache đã được xóa khẩn cấp"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency clear cache failed: {str(e)}")

#==========================================================================================================================================
# HELPER FUNCTIONS
#==========================================================================================================================================

def _get_health_recommendations(
    agents_health: Dict,
    db_health: Dict,
    storage_health: Dict,
    external_health: Dict
) -> List[str]:
    """
    Tạo recommendations dựa trên tình trạng system health
    """
    recommendations = []

    # Kiểm tra agents health
    if agents_health.get("overall_status") != "healthy":
        recommendations.append("Kiểm tra và restart các agents bị lỗi")

    # Kiểm tra database health
    if db_health.get("status") != "healthy":
        recommendations.append("Kiểm tra kết nối database và performance")

    # Kiểm tra storage health
    storage_usage = storage_health.get("usage_percent", 0)
    if storage_usage > 80:
        recommendations.append("Dung lượng storage cao (>80%), cần dọn dẹp hoặc mở rộng")
    elif storage_usage > 90:
        recommendations.append("CẢNH BÁO: Dung lượng storage rất cao (>90%), cần xử lý ngay")

    # Kiểm tra external services
    if external_health.get("status") != "healthy":
        recommendations.append("Kiểm tra kết nối với external services")

    # Nếu không có vấn đề gì
    if not recommendations:
        recommendations.append("Hệ thống đang hoạt động bình thường")

    return recommendations

#==========================================================================================================================================
# HEALTH CHECK
#==========================================================================================================================================

@router.get("/health")
async def admin_health():
    """
    Health check cho admin endpoints
    """
    try:
        # Test database connection
        db_ok = await admin_service.test_database_connection()

        # Test agents connection
        agents_ok = agents_manager.get_system_health_summary().get("overall_status") == "healthy"

        return {
            "status": "healthy" if (db_ok and agents_ok) else "degraded",
            "database_connection": "ok" if db_ok else "error",
            "agents_connection": "ok" if agents_ok else "error",
            "endpoints": [
                "/admin/dashboard",
                "/admin/users",
                "/admin/videos",
                "/admin/system/health",
                "/admin/actions/log"
            ],
            "admin_features": [
                "user_management",
                "video_management",
                "system_monitoring",
                "emergency_controls",
                "audit_logging"
            ]
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }