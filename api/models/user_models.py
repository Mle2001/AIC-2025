# api/models/user_models.py
"""
User Models - Pydantic models cho user management system
Dev2: API Data Models - định nghĩa structure cho user requests/responses
Current: 2025-07-03 14:19:00 UTC, User: xthanh1910
"""

from pydantic import BaseModel, Field, validator, EmailStr, root_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import re

# ================================
# ENUMS CHO USER SYSTEM
# ================================

class UserRole(str, Enum):
    """
    Vai trò của user trong hệ thống
    """
    ADMIN = "admin"               # Quản trị viên hệ thống
    MODERATOR = "moderator"       # Người kiểm duyệt
    USER = "user"                # User thường
    PREMIUM = "premium"          # User premium
    ENTERPRISE = "enterprise"    # User doanh nghiệp
    GUEST = "guest"              # Khách (limited access)

class UserStatus(str, Enum):
    """
    Trạng thái tài khoản user
    """
    ACTIVE = "active"            # Tài khoản hoạt động
    INACTIVE = "inactive"        # Tài khoản không hoạt động
    PENDING = "pending"          # Đang chờ xác nhận
    SUSPENDED = "suspended"      # Tạm dừng
    BANNED = "banned"           # Bị cấm
    DELETED = "deleted"         # Đã xóa

class ActivityType(str, Enum):
    """
    Loại hoạt động của user
    """
    LOGIN = "login"                    # Đăng nhập
    LOGOUT = "logout"                  # Đăng xuất
    REGISTER = "register"              # Đăng ký
    PROFILE_UPDATE = "profile_update"  # Cập nhật profile
    PASSWORD_CHANGE = "password_change" # Đổi mật khẩu
    VIDEO_UPLOAD = "video_upload"      # Upload video
    VIDEO_VIEW = "video_view"          # Xem video
    CHAT = "chat"                     # Chat với AI
    SEARCH = "search"                 # Tìm kiếm
    ADMIN_ACTION = "admin_action"     # Hành động admin

class PrivacyLevel(str, Enum):
    """
    Mức độ riêng tư
    """
    PUBLIC = "public"            # Công khai
    FRIENDS = "friends"          # Bạn bè
    PRIVATE = "private"          # Riêng tư

class NotificationType(str, Enum):
    """
    Loại thông báo
    """
    SYSTEM = "system"            # Thông báo hệ thống
    PROCESSING = "processing"    # Thông báo về xử lý video
    SOCIAL = "social"           # Thông báo xã hội
    SECURITY = "security"       # Thông báo bảo mật
    MARKETING = "marketing"     # Thông báo marketing

# ================================
# USER AUTHENTICATION MODELS
# ================================

class UserRegisterRequest(BaseModel):
    """
    Request đăng ký user mới
    """
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        regex="^[a-zA-Z0-9_]+$",
        description="Tên đăng nhập (chỉ chữ, số và _)"
    )
    email: EmailStr = Field(..., description="Email của user")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Mật khẩu (ít nhất 8 ký tự)"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Họ và tên đầy đủ"
    )
    agree_terms: bool = Field(..., description="Đồng ý với điều khoản sử dụng")
    referral_code: Optional[str] = Field(None, description="Mã giới thiệu")

    @validator('username')
    def validate_username(cls, v):
        """
        Validate username
        """
        v = v.lower().strip()

        # Kiểm tra từ cấm
        forbidden_usernames = [
            'admin', 'administrator', 'root', 'system', 'api', 'www',
            'mail', 'ftp', 'support', 'help', 'info', 'test', 'demo'
        ]
        if v in forbidden_usernames:
            raise ValueError('Username này không được phép sử dụng')

        # Không được bắt đầu bằng số
        if v[0].isdigit():
            raise ValueError('Username không được bắt đầu bằng số')

        return v

    @validator('password')
    def validate_password(cls, v):
        """
        Validate password strength
        """
        # Kiểm tra có ít nhất 1 chữ hoa
        if not re.search(r'[A-Z]', v):
            raise ValueError('Mật khẩu phải có ít nhất 1 chữ hoa')

        # Kiểm tra có ít nhất 1 chữ thường
        if not re.search(r'[a-z]', v):
            raise ValueError('Mật khẩu phải có ít nhất 1 chữ thường')

        # Kiểm tra có ít nhất 1 số
        if not re.search(r'\d', v):
            raise ValueError('Mật khẩu phải có ít nhất 1 chữ số')

        # Kiểm tra có ít nhất 1 ký tự đặc biệt
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Mật khẩu phải có ít nhất 1 ký tự đặc biệt')

        return v

    @validator('full_name')
    def validate_full_name(cls, v):
        """
        Validate full name
        """
        v = v.strip()

        # Kiểm tra không chứa số
        if re.search(r'\d', v):
            raise ValueError('Họ tên không được chứa số')

        # Kiểm tra không chứa ký tự đặc biệt (trừ space, dấu tiếng Việt)
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Họ tên chứa ký tự không hợp lệ')

        return v

    @validator('agree_terms')
    def validate_terms(cls, v):
        """
        Validate terms agreement
        """
        if not v:
            raise ValueError('Bạn phải đồng ý với điều khoản sử dụng')
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "nguyen_van_a",
                "email": "nguyen.van.a@example.com",
                "password": "MySecure123!",
                "full_name": "Nguyễn Văn A",
                "agree_terms": True,
                "referral_code": "REF123"
            }
        }

class UserLoginRequest(BaseModel):
    """
    Request đăng nhập user
    """
    username_or_email: str = Field(..., description="Username hoặc email")
    password: str = Field(..., description="Mật khẩu")
    remember_me: bool = Field(False, description="Ghi nhớ đăng nhập")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Thông tin thiết bị")

    @validator('username_or_email')
    def validate_username_or_email(cls, v):
        v = v.strip().lower()
        if not v:
            raise ValueError('Username hoặc email không được để trống')
        return v

    class Config:
        schema_extra = {
            "example": {
                "username_or_email": "nguyen_van_a",
                "password": "MySecure123!",
                "remember_me": True,
                "device_info": {
                    "device_type": "web",
                    "browser": "Chrome",
                    "os": "Windows 10"
                }
            }
        }

class UserAuthResponse(BaseModel):
    """
    Response đăng nhập thành công
    """
    success: bool = Field(..., description="Đăng nhập có thành công không")
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("Bearer", description="Loại token")
    expires_in: int = Field(..., description="Thời gian hết hạn (seconds)")
    user: 'UserBasicInfo' = Field(..., description="Thông tin user cơ bản")
    session_id: str = Field(..., description="Session ID")
    permissions: List[str] = Field([], description="Danh sách quyền")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "Bearer",
                "expires_in": 86400,
                "user": {
                    "user_id": "user_123",
                    "username": "nguyen_van_a",
                    "email": "nguyen.van.a@example.com",
                    "full_name": "Nguyễn Văn A",
                    "role": "user",
                    "avatar_url": None
                },
                "session_id": "session_456",
                "permissions": ["upload_video", "chat", "search"]
            }
        }

class PasswordChangeRequest(BaseModel):
    """
    Request đổi mật khẩu
    """
    current_password: str = Field(..., description="Mật khẩu hiện tại")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Mật khẩu mới"
    )
    confirm_password: str = Field(..., description="Xác nhận mật khẩu mới")

    @validator('new_password')
    def validate_new_password(cls, v):
        """
        Validate new password strength (same as register)
        """
        if not re.search(r'[A-Z]', v):
            raise ValueError('Mật khẩu mới phải có ít nhất 1 chữ hoa')
        if not re.search(r'[a-z]', v):
            raise ValueError('Mật khẩu mới phải có ít nhất 1 chữ thường')
        if not re.search(r'\d', v):
            raise ValueError('Mật khẩu mới phải có ít nhất 1 chữ số')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Mật khẩu mới phải có ít nhất 1 ký tự đặc biệt')
        return v

    @root_validator
    def validate_passwords_match(cls, values):
        """
        Validate passwords match
        """
        new_password = values.get('new_password')
        confirm_password = values.get('confirm_password')

        if new_password and confirm_password and new_password != confirm_password:
            raise ValueError('Mật khẩu xác nhận không trùng khớp')

        current_password = values.get('current_password')
        if current_password and new_password and current_password == new_password:
            raise ValueError('Mật khẩu mới phải khác mật khẩu hiện tại')

        return values

    class Config:
        schema_extra = {
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecure456@",
                "confirm_password": "NewSecure456@"
            }
        }

# ================================
# USER PROFILE MODELS
# ================================

class UserPreferences(BaseModel):
    """
    Preferences của user
    """
    theme: str = Field("light", regex="^(light|dark|auto)$", description="Theme giao diện")
    language: str = Field("vi", regex="^(vi|en)$", description="Ngôn ngữ")
    timezone: str = Field("Asia/Ho_Chi_Minh", description="Múi giờ")
    notifications_enabled: bool = Field(True, description="Bật thông báo")
    email_notifications: bool = Field(True, description="Thông báo qua email")
    auto_play_videos: bool = Field(True, description="Tự động phát video")
    video_quality: str = Field("auto", regex="^(auto|360p|480p|720p|1080p)$", description="Chất lượng video")
    search_suggestions: bool = Field(True, description="Hiển thị gợi ý tìm kiếm")

    class Config:
        schema_extra = {
            "example": {
                "theme": "dark",
                "language": "vi",
                "timezone": "Asia/Ho_Chi_Minh",
                "notifications_enabled": True,
                "email_notifications": False,
                "auto_play_videos": True,
                "video_quality": "1080p",
                "search_suggestions": True
            }
        }

class UserSettings(BaseModel):
    """
    Settings của user
    """
    privacy_level: PrivacyLevel = Field(PrivacyLevel.PUBLIC, description="Mức độ riêng tư")
    allow_contact: bool = Field(True, description="Cho phép liên hệ")
    show_activity: bool = Field(True, description="Hiển thị hoạt động")
    searchable: bool = Field(True, description="Có thể tìm kiếm được")
    two_factor_enabled: bool = Field(False, description="Bật xác thực 2 bước")
    session_timeout: int = Field(1440, description="Timeout session (phút)")
    data_retention_days: int = Field(365, description="Lưu trữ dữ liệu (ngày)")

    class Config:
        schema_extra = {
            "example": {
                "privacy_level": "public",
                "allow_contact": True,
                "show_activity": False,
                "searchable": True,
                "two_factor_enabled": True,
                "session_timeout": 720,
                "data_retention_days": 365
            }
        }

class UserProfile(BaseModel):
    """
    Profile đầy đủ của user
    """
    user_id: str = Field(..., description="ID của user")
    username: str = Field(..., description="Tên đăng nhập")
    email: EmailStr = Field(..., description="Email")
    full_name: str = Field(..., description="Họ và tên")
    avatar_url: Optional[str] = Field(None, description="URL avatar")
    bio: Optional[str] = Field(None, max_length=500, description="Tiểu sử")
    website: Optional[str] = Field(None, description="Website cá nhân")
    location: Optional[str] = Field(None, description="Vị trí")
    birth_date: Optional[datetime] = Field(None, description="Ngày sinh")
    phone: Optional[str] = Field(None, description="Số điện thoại")

    # System info
    role: UserRole = Field(..., description="Vai trò trong hệ thống")
    status: UserStatus = Field(..., description="Trạng thái tài khoản")
    is_verified: bool = Field(..., description="Đã xác minh email")
    created_at: datetime = Field(..., description="Ngày tạo tài khoản")
    updated_at: datetime = Field(..., description="Ngày cập nhật cuối")
    last_login: Optional[datetime] = Field(None, description="Lần đăng nhập cuối")

    # User preferences and settings
    preferences: UserPreferences = Field(..., description="Preferences của user")
    settings: UserSettings = Field(..., description="Settings của user")

    # Statistics
    video_count: int = Field(0, description="Số video đã upload")
    total_views: int = Field(0, description="Tổng lượt xem videos")
    chat_sessions: int = Field(0, description="Số session chat")
    last_activity: Optional[datetime] = Field(None, description="Hoạt động cuối")

    @validator('website')
    def validate_website(cls, v):
        """
        Validate website URL
        """
        if v:
            v = v.strip()
            if not v.startswith(('http://', 'https://')):
                v = 'https://' + v

            # Basic URL validation
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', v):
                raise ValueError('URL website không hợp lệ')

        return v

    @validator('phone')
    def validate_phone(cls, v):
        """
        Validate phone number
        """
        if v:
            # Remove all non-digits
            digits_only = re.sub(r'\D', '', v)

            # Vietnam phone validation
            if not re.match(r'^(84|0)[3|5|7|8|9][0-9]{8}$', digits_only):
                raise ValueError('Số điện thoại không hợp lệ')

            return digits_only

        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "nguyen_van_a",
                "email": "nguyen.van.a@example.com",
                "full_name": "Nguyễn Văn A",
                "avatar_url": "https://example.com/avatar/user_123.jpg",
                "bio": "Đầu bếp chuyên nghiệp, yêu thích chia sẻ công thức nấu ăn",
                "website": "https://chefnguyenvana.com",
                "location": "Hồ Chí Minh, Việt Nam",
                "birth_date": "1990-05-15T00:00:00Z",
                "phone": "0901234567",
                "role": "user",
                "status": "active",
                "is_verified": True,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-07-03T14:19:00Z",
                "last_login": "2025-07-03T14:15:00Z",
                "preferences": {
                    "theme": "dark",
                    "language": "vi"
                },
                "settings": {
                    "privacy_level": "public",
                    "allow_contact": True
                },
                "video_count": 25,
                "total_views": 15420,
                "chat_sessions": 89,
                "last_activity": "2025-07-03T14:18:00Z"
            }
        }

class UserBasicInfo(BaseModel):
    """
    Thông tin user cơ bản (cho responses, references)
    """
    user_id: str = Field(..., description="ID của user")
    username: str = Field(..., description="Tên đăng nhập")
    email: Optional[EmailStr] = Field(None, description="Email (optional)")
    full_name: str = Field(..., description="Họ và tên")
    avatar_url: Optional[str] = Field(None, description="URL avatar")
    role: UserRole = Field(..., description="Vai trò")
    status: UserStatus = Field(..., description="Trạng thái")
    is_verified: bool = Field(..., description="Đã xác minh")
    last_activity: Optional[datetime] = Field(None, description="Hoạt động cuối")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "nguyen_van_a",
                "email": "nguyen.van.a@example.com",
                "full_name": "Nguyễn Văn A",
                "avatar_url": "https://example.com/avatar.jpg",
                "role": "user",
                "status": "active",
                "is_verified": True,
                "last_activity": "2025-07-03T14:18:00Z"
            }
        }

class UserUpdateRequest(BaseModel):
    """
    Request cập nhật profile user
    """
    full_name: Optional[str] = Field(None, min_length=2, max_length=100, description="Họ và tên")
    bio: Optional[str] = Field(None, max_length=500, description="Tiểu sử")
    website: Optional[str] = Field(None, description="Website")
    location: Optional[str] = Field(None, description="Vị trí")
    birth_date: Optional[datetime] = Field(None, description="Ngày sinh")
    phone: Optional[str] = Field(None, description="Số điện thoại")
    preferences: Optional[UserPreferences] = Field(None, description="Preferences")
    settings: Optional[UserSettings] = Field(None, description="Settings")

    @validator('website')
    def validate_website(cls, v):
        if v:
            v = v.strip()
            if not v.startswith(('http://', 'https://')):
                v = 'https://' + v
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', v):
                raise ValueError('URL website không hợp lệ')
        return v

    @validator('phone')
    def validate_phone(cls, v):
        if v:
            digits_only = re.sub(r'\D', '', v)
            if not re.match(r'^(84|0)[3|5|7|8|9][0-9]{8}$', digits_only):
                raise ValueError('Số điện thoại không hợp lệ')
            return digits_only
        return v

    @validator('birth_date')
    def validate_birth_date(cls, v):
        if v:
            # Kiểm tra tuổi hợp lệ (13-120 tuổi)
            now = datetime.utcnow()
            age = (now - v).days / 365.25

            if age < 13:
                raise ValueError('Bạn phải ít nhất 13 tuổi để sử dụng dịch vụ')
            if age > 120:
                raise ValueError('Ngày sinh không hợp lệ')

        return v

    class Config:
        schema_extra = {
            "example": {
                "full_name": "Nguyễn Văn A",
                "bio": "Đầu bếp chuyên nghiệp",
                "website": "https://chefnguyenvana.com",
                "location": "Hồ Chí Minh",
                "phone": "0901234567",
                "preferences": {
                    "theme": "dark",
                    "language": "vi"
                },
                "settings": {
                    "privacy_level": "public"
                }
            }
        }

# ================================
# USER ACTIVITY MODELS
# ================================

class UserActivity(BaseModel):
    """
    Hoạt động của user
    """
    activity_id: str = Field(..., description="ID của activity")
    user_id: str = Field(..., description="ID của user")
    activity_type: ActivityType = Field(..., description="Loại hoạt động")
    activity_data: Dict[str, Any] = Field(..., description="Dữ liệu hoạt động")
    ip_address: Optional[str] = Field(None, description="Địa chỉ IP")
    user_agent: Optional[str] = Field(None, description="User agent")
    timestamp: datetime = Field(..., description="Thời gian hoạt động")

    # Additional context
    session_id: Optional[str] = Field(None, description="Session ID")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Thông tin thiết bị")
    location_info: Optional[Dict[str, Any]] = Field(None, description="Thông tin vị trí")

    class Config:
        schema_extra = {
            "example": {
                "activity_id": "activity_123",
                "user_id": "user_456",
                "activity_type": "video_upload",
                "activity_data": {
                    "video_id": "video_789",
                    "filename": "cooking_tutorial.mp4",
                    "file_size": 157286400
                },
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "timestamp": "2025-07-03T14:19:00Z",
                "session_id": "session_abc",
                "device_info": {
                    "device_type": "desktop",
                    "browser": "Chrome",
                    "os": "Windows 10"
                }
            }
        }

class UserActivitySummary(BaseModel):
    """
    Tóm tắt hoạt động của user
    """
    user_id: str = Field(..., description="ID của user")
    total_activities: int = Field(..., description="Tổng số hoạt động")
    activities_by_type: Dict[str, int] = Field(..., description="Hoạt động theo loại")
    last_activity: Optional[datetime] = Field(None, description="Hoạt động cuối")
    most_active_hour: Optional[int] = Field(None, description="Giờ hoạt động nhiều nhất")
    devices_used: List[str] = Field([], description="Thiết bị đã sử dụng")
    total_sessions: int = Field(0, description="Tổng số sessions")
    avg_session_duration: Optional[float] = Field(None, description="Thời lượng session trung bình")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_456",
                "total_activities": 245,
                "activities_by_type": {
                    "login": 89,
                    "video_upload": 25,
                    "chat": 67,
                    "search": 64
                },
                "last_activity": "2025-07-03T14:18:00Z",
                "most_active_hour": 14,
                "devices_used": ["desktop", "mobile"],
                "total_sessions": 89,
                "avg_session_duration": 25.5
            }
        }

# ================================
# ADMIN USER MANAGEMENT MODELS
# ================================

class AdminUser(BaseModel):
    """
    Admin user với permissions mở rộng
    """
    user_id: str = Field(..., description="ID của admin")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email")
    full_name: str = Field(..., description="Họ và tên")
    role: UserRole = Field(..., description="Role (admin/moderator)")
    permissions: List[str] = Field(..., description="Danh sách permissions")
    is_super_admin: bool = Field(False, description="Có phải super admin không")
    admin_level: int = Field(1, description="Cấp độ admin (1-10)")
    last_admin_activity: Optional[datetime] = Field(None, description="Hoạt động admin cuối")

    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role in [UserRole.ADMIN, UserRole.MODERATOR]

    def has_permission(self, permission: str) -> bool:
        """Check if admin has specific permission"""
        return permission in self.permissions or self.is_super_admin

    class Config:
        schema_extra = {
            "example": {
                "user_id": "admin_xthanh1910",
                "username": "xthanh1910",
                "email": "xthanh1910@admin.com",
                "full_name": "Admin User",
                "role": "admin",
                "permissions": [
                    "user_management",
                    "video_management",
                    "system_admin",
                    "emergency_controls"
                ],
                "is_super_admin": True,
                "admin_level": 10,
                "last_admin_activity": "2025-07-03T14:19:00Z"
            }
        }

class UserManagementResponse(BaseModel):
    """
    Response cho admin user management
    """
    success: bool = Field(..., description="Thao tác có thành công không")
    total_users: int = Field(..., description="Tổng số users")
    page: int = Field(..., description="Trang hiện tại")
    limit: int = Field(..., description="Số users mỗi trang")
    users: List[UserProfile] = Field(..., description="Danh sách users")
    filters_applied: Dict[str, Any] = Field(..., description="Filters đã áp dụng")
    admin_action_by: str = Field(..., description="Admin thực hiện action")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_users": 1250,
                "page": 1,
                "limit": 50,
                "users": [],
                "filters_applied": {
                    "status": "active",
                    "role": "user"
                },
                "admin_action_by": "xthanh1910"
            }
        }

class UserStatusUpdateRequest(BaseModel):
    """
    Request cập nhật status user (admin only)
    """
    new_status: UserStatus = Field(..., description="Status mới")
    reason: str = Field(..., min_length=10, max_length=500, description="Lý do thay đổi")
    notify_user: bool = Field(True, description="Thông báo cho user")
    duration_days: Optional[int] = Field(None, description="Thời gian (cho suspend/ban)")

    @validator('duration_days')
    def validate_duration(cls, v, values):
        new_status = values.get('new_status')
        if new_status in [UserStatus.SUSPENDED, UserStatus.BANNED] and not v:
            raise ValueError('Phải có thời gian cho suspend/ban')
        if v and v > 365:
            raise ValueError('Thời gian tối đa 365 ngày')
        return v

    class Config:
        schema_extra = {
            "example": {
                "new_status": "suspended",
                "reason": "Vi phạm quy định cộng đồng - spam nội dung",
                "notify_user": True,
                "duration_days": 7
            }
        }

# ================================
# USER STATISTICS MODELS
# ================================

class UserStatistics(BaseModel):
    """
    Thống kê user system
    """
    total_users: int = Field(..., description="Tổng số users")
    active_users: int = Field(..., description="Users đang hoạt động")
    new_users_today: int = Field(..., description="Users mới hôm nay")
    new_users_this_week: int = Field(..., description="Users mới tuần này")
    new_users_this_month: int = Field(..., description="Users mới tháng này")

    users_by_status: Dict[str, int] = Field(..., description="Users theo status")
    users_by_role: Dict[str, int] = Field(..., description="Users theo role")

    # Activity stats
    total_sessions_today: int = Field(..., description="Sessions hôm nay")
    avg_session_duration: float = Field(..., description="Thời lượng session trung bình")
    most_active_users: List[Dict[str, Any]] = Field(..., description="Users hoạt động nhiều nhất")

    # Geographic stats
    users_by_country: Dict[str, int] = Field(..., description="Users theo quốc gia")
    users_by_timezone: Dict[str, int] = Field(..., description="Users theo timezone")

    class Config:
        schema_extra = {
            "example": {
                "total_users": 12500,
                "active_users": 8945,
                "new_users_today": 45,
                "new_users_this_week": 289,
                "new_users_this_month": 1250,
                "users_by_status": {
                    "active": 11200,
                    "inactive": 950,
                    "suspended": 250,
                    "banned": 100
                },
                "users_by_role": {
                    "user": 12450,
                    "premium": 45,
                    "admin": 5
                },
                "total_sessions_today": 2345,
                "avg_session_duration": 28.5,
                "most_active_users": [
                    {
                        "user_id": "user_123",
                        "username": "power_user",
                        "activity_count": 156
                    }
                ],
                "users_by_country": {
                    "VN": 8900,
                    "US": 1200,
                    "JP": 800
                },
                "users_by_timezone": {
                    "Asia/Ho_Chi_Minh": 8900,
                    "America/New_York": 1200
                }
            }
        }

# ================================
# NOTIFICATION MODELS
# ================================

class UserNotification(BaseModel):
    """
    Thông báo cho user
    """
    notification_id: str = Field(..., description="ID thông báo")
    user_id: str = Field(..., description="ID người nhận")
    type: NotificationType = Field(..., description="Loại thông báo")
    title: str = Field(..., description="Tiêu đề thông báo")
    message: str = Field(..., description="Nội dung thông báo")
    data: Optional[Dict[str, Any]] = Field(None, description="Dữ liệu bổ sung")

    is_read: bool = Field(False, description="Đã đọc chưa")
    is_important: bool = Field(False, description="Quan trọng không")
    created_at: datetime = Field(..., description="Thời gian tạo")
    read_at: Optional[datetime] = Field(None, description="Thời gian đọc")
    expires_at: Optional[datetime] = Field(None, description="Thời gian hết hạn")

    class Config:
        schema_extra = {
            "example": {
                "notification_id": "notif_123",
                "user_id": "user_456",
                "type": "processing",
                "title": "Video processing completed",
                "message": "Your video 'Cooking Tutorial' has been processed successfully",
                "data": {
                    "video_id": "video_789",
                    "video_title": "Cooking Tutorial"
                },
                "is_read": False,
                "is_important": True,
                "created_at": "2025-07-03T14:19:00Z",
                "read_at": None,
                "expires_at": "2025-07-10T14:19:00Z"
            }
        }

# ================================
# UTILITY MODELS
# ================================

class User(UserProfile):
    """
    User model đầy đủ (alias cho UserProfile)
    Để compatibility với code cũ
    """
    pass

class APIResponse(BaseModel):
    """
    Generic API response
    """
    success: bool = Field(..., description="Request có thành công không")
    message: str = Field(..., description="Thông báo")
    data: Optional[Any] = Field(None, description="Dữ liệu response")
    error_code: Optional[str] = Field(None, description="Mã lỗi")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Thời gian response")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Profile updated successfully",
                "data": {"user_id": "user_123"},
                "error_code": None,
                "timestamp": "2025-07-03T14:19:00Z"
            }
        }

# ================================
# UTILITY FUNCTIONS
# ================================

def create_user_response(user_profile: UserProfile) -> APIResponse:
    """
    Helper function tạo user response
    """
    return APIResponse(
        success=True,
        message="User data retrieved successfully",
        data=user_profile.dict()
    )

def create_auth_response(
    access_token: str,
    user: UserBasicInfo,
    session_id: str,
    expires_in: int = 86400
) -> UserAuthResponse:
    """
    Helper function tạo auth response
    """
    return UserAuthResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        user=user,
        session_id=session_id,
        permissions=_get_user_permissions(user.role)
    )

def _get_user_permissions(role: UserRole) -> List[str]:
    """
    Get permissions theo role
    """
    permissions_map = {
        UserRole.GUEST: ["search", "view_public"],
        UserRole.USER: ["search", "view_public", "upload_video", "chat", "profile_edit"],
        UserRole.PREMIUM: ["search", "view_public", "upload_video", "chat", "profile_edit", "priority_processing"],
        UserRole.MODERATOR: ["search", "view_public", "upload_video", "chat", "profile_edit", "moderate_content"],
        UserRole.ADMIN: ["all_permissions"],
        UserRole.ENTERPRISE: ["search", "view_public", "upload_video", "chat", "profile_edit", "bulk_operations", "api_access"]
    }
    return permissions_map.get(role, [])

# ================================
# FORWARD REFERENCES
# ================================

# Update forward references
UserAuthResponse.model_rebuild()

# ================================
# EXPORTS
# ================================

__all__ = [
    # Enums
    "UserRole",
    "UserStatus",
    "ActivityType",
    "PrivacyLevel",
    "NotificationType",

    # Auth Models
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserAuthResponse",
    "PasswordChangeRequest",

    # Profile Models
    "UserProfile",
    "UserBasicInfo",
    "UserUpdateRequest",
    "UserPreferences",
    "UserSettings",

    # Activity Models
    "UserActivity",
    "UserActivitySummary",

    # Admin Models
    "AdminUser",
    "UserManagementResponse",
    "UserStatusUpdateRequest",

    # Statistics Models
    "UserStatistics",

    # Notification Models
    "UserNotification",

    # Utility Models
    "User",
    "APIResponse",

    # Helper Functions
    "create_user_response",
    "create_auth_response"
]