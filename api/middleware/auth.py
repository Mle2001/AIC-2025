# api/middleware/auth.py
"""
Authentication Middleware - JWT authentication và authorization cho API
Dev2: API Security & Middleware - bảo vệ endpoints và quản lý permissions
Current: 2025-07-03 14:21:32 UTC, User: xthanh1910
"""

import jwt
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from functools import wraps

from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

# Import models và services
from ..models.user_models import User, AdminUser, UserRole, UserStatus
from ..services.user_service import UserService
from ..services.cache_service import CacheService

# Khởi tạo security và services
security = HTTPBearer()
user_service = UserService()
cache_service = CacheService()

# ================================
# AUTHENTICATION CONFIGURATION
# ================================

class AuthConfig:
    """
    Configuration cho authentication system
    """
    # JWT settings
    JWT_SECRET_KEY = "dev_secret_key_change_in_production_xthanh1910"  # Dev3 sẽ cung cấp từ env
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_HOURS = 24

    # Security settings
    REQUIRE_EMAIL_VERIFICATION = False  # Set True trong production
    ALLOW_MULTIPLE_SESSIONS = True
    SESSION_TIMEOUT_MINUTES = 1440  # 24 hours

    # Rate limiting
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

    # Admin settings
    ADMIN_SESSION_TIMEOUT_MINUTES = 480  # 8 hours for admins
    REQUIRE_ADMIN_2FA = False  # Set True trong production

    # Cache settings
    CACHE_USER_DATA_TTL = 3600  # 1 hour
    CACHE_PERMISSIONS_TTL = 1800  # 30 minutes

auth_config = AuthConfig()

# ================================
# EXCEPTION CLASSES
# ================================

class AuthenticationError(HTTPException):
    """
    Custom authentication error
    """
    def __init__(self, detail: str, error_code: str = "AUTH_ERROR"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer", "X-Error-Code": error_code}
        )

class AuthorizationError(HTTPException):
    """
    Custom authorization error
    """
    def __init__(self, detail: str, error_code: str = "AUTHORIZATION_ERROR"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers={"X-Error-Code": error_code}
        )

class SessionExpiredError(AuthenticationError):
    """
    Session expired error
    """
    def __init__(self):
        super().__init__(
            detail="Session đã hết hạn, vui lòng đăng nhập lại",
            error_code="SESSION_EXPIRED"
        )

class InsufficientPermissionsError(AuthorizationError):
    """
    Insufficient permissions error
    """
    def __init__(self, required_permission: str):
        super().__init__(
            detail=f"Không có quyền {required_permission}",
            error_code="INSUFFICIENT_PERMISSIONS"
        )

# ================================
# JWT TOKEN UTILITIES
# ================================

class TokenManager:
    """
    Manager cho JWT token operations
    """

    @staticmethod
    def create_access_token(user_data: Dict[str, Any]) -> str:
        """
        Tạo JWT access token
        """
        try:
            # Payload cho JWT
            payload = {
                "user_id": user_data["user_id"],
                "username": user_data["username"],
                "email": user_data.get("email"),
                "role": user_data.get("role", "user"),
                "status": user_data.get("status", "active"),
                "is_verified": user_data.get("is_verified", False),
                "iat": int(time.time()),  # Issued at
                "exp": int(time.time()) + (auth_config.JWT_EXPIRE_HOURS * 3600),  # Expires at
                "iss": "ai_challenge_api",  # Issuer
                "aud": "ai_challenge_users"  # Audience
            }

            # Encode JWT token
            token = jwt.encode(
                payload,
                auth_config.JWT_SECRET_KEY,
                algorithm=auth_config.JWT_ALGORITHM
            )

            print(f"[{datetime.utcnow()}] JWT token created for user: {user_data['username']}")
            return token

        except Exception as e:
            print(f"Error creating JWT token: {str(e)}")
            raise AuthenticationError("Không thể tạo token xác thực")

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode và validate JWT token
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                auth_config.JWT_SECRET_KEY,
                algorithms=[auth_config.JWT_ALGORITHM],
                audience="ai_challenge_users",
                issuer="ai_challenge_api"
            )

            return payload

        except jwt.ExpiredSignatureError:
            raise SessionExpiredError()
        except jwt.InvalidTokenError as e:
            print(f"Invalid JWT token: {str(e)}")
            raise AuthenticationError("Token không hợp lệ", "INVALID_TOKEN")
        except Exception as e:
            print(f"Token decode error: {str(e)}")
            raise AuthenticationError("Lỗi xác thực token", "TOKEN_DECODE_ERROR")

    @staticmethod
    def refresh_token(token: str) -> str:
        """
        Refresh JWT token nếu sắp hết hạn
        """
        try:
            payload = TokenManager.decode_token(token)

            # Kiểm tra token có gần hết hạn không (còn < 2 hours)
            exp_time = payload.get("exp", 0)
            current_time = int(time.time())
            time_remaining = exp_time - current_time

            if time_remaining < 7200:  # < 2 hours
                # Tạo token mới
                user_data = {
                    "user_id": payload["user_id"],
                    "username": payload["username"],
                    "email": payload.get("email"),
                    "role": payload.get("role"),
                    "status": payload.get("status"),
                    "is_verified": payload.get("is_verified")
                }

                return TokenManager.create_access_token(user_data)

            return token  # Token còn mới, không cần refresh

        except Exception as e:
            print(f"Token refresh error: {str(e)}")
            raise AuthenticationError("Không thể refresh token")

# ================================
# USER AUTHENTICATION FUNCTIONS
# ================================

async def get_token_from_header(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Extract token từ Authorization header
    """
    if not credentials:
        raise AuthenticationError("Thiếu token xác thực", "MISSING_TOKEN")

    if credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Token phải có format Bearer", "INVALID_TOKEN_FORMAT")

    return credentials.credentials

async def get_current_user_from_token(token: str) -> User:
    """
    Lấy user hiện tại từ JWT token
    """
    try:
        # Bước 1: Decode token
        payload = TokenManager.decode_token(token)
        user_id = payload.get("user_id")

        if not user_id:
            raise AuthenticationError("Token không chứa user_id", "INVALID_TOKEN_PAYLOAD")

        # Bước 2: Thử lấy user từ cache trước
        cache_key = f"user_auth:{user_id}"
        cached_user = await cache_service.get(cache_key, namespace='user')

        if cached_user:
            # Verify token vẫn còn hợp lệ trong cache
            cached_token = cached_user.get("access_token")
            if cached_token == token:
                return User(**cached_user["user_data"])

        # Bước 3: Lấy user từ database thông qua service
        user_data = await user_service.get_user_profile(user_id)

        if not user_data:
            raise AuthenticationError("User không tồn tại", "USER_NOT_FOUND")

        # Bước 4: Kiểm tra user status
        if user_data.get("status") != UserStatus.ACTIVE:
            user_status = user_data.get("status", "unknown")
            if user_status == UserStatus.BANNED:
                raise AuthenticationError("Tài khoản đã bị cấm", "ACCOUNT_BANNED")
            elif user_status == UserStatus.SUSPENDED:
                raise AuthenticationError("Tài khoản tạm thời bị đình chỉ", "ACCOUNT_SUSPENDED")
            elif user_status == UserStatus.INACTIVE:
                raise AuthenticationError("Tài khoản chưa được kích hoạt", "ACCOUNT_INACTIVE")
            else:
                raise AuthenticationError(f"Tài khoản {user_status}", "ACCOUNT_STATUS_ERROR")

        # Bước 5: Kiểm tra email verification nếu required
        if auth_config.REQUIRE_EMAIL_VERIFICATION and not user_data.get("is_verified"):
            raise AuthenticationError("Email chưa được xác minh", "EMAIL_NOT_VERIFIED")

        # Bước 6: Cache user data
        cache_data = {
            "user_data": user_data,
            "access_token": token,
            "cached_at": time.time()
        }
        await cache_service.set(
            cache_key,
            cache_data,
            ttl=auth_config.CACHE_USER_DATA_TTL,
            namespace='user'
        )

        # Bước 7: Update last activity
        await user_service.track_user_activity(
            user_id=user_id,
            activity_type="api_access",
            activity_data={"endpoint": "auth_check", "token_used": True}
        )

        print(f"[{datetime.utcnow()}] User authenticated: {user_data['username']} (ID: {user_id})")
        return User(**user_data)

    except HTTPException:
        # Re-raise HTTP exceptions (đã được handle)
        raise
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        raise AuthenticationError("Lỗi xác thực người dùng", "AUTH_SYSTEM_ERROR")

async def get_current_user(token: str = Depends(get_token_from_header)) -> User:
    """
    Dependency để lấy current user (cho protected endpoints)
    """
    return await get_current_user_from_token(token)

async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> AdminUser:
    """
    Dependency để lấy current admin user (chỉ admin mới access được)
    """
    try:
        # Kiểm tra user có phải admin không
        if current_user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise AuthorizationError(
                "Chỉ admin mới có quyền truy cập",
                "ADMIN_REQUIRED"
            )

        # Lấy admin permissions từ cache hoặc database
        cache_key = f"admin_permissions:{current_user.user_id}"
        cached_permissions = await cache_service.get(cache_key, namespace='user')

        if not cached_permissions:
            # Load admin permissions (sẽ implement khi có database)
            permissions = _get_admin_permissions(current_user.role)
            await cache_service.set(
                cache_key,
                permissions,
                ttl=auth_config.CACHE_PERMISSIONS_TTL,
                namespace='user'
            )
        else:
            permissions = cached_permissions

        # Tạo AdminUser object
        admin_user = AdminUser(
            user_id=current_user.user_id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            role=current_user.role,
            permissions=permissions,
            is_super_admin=(current_user.role == UserRole.ADMIN),
            admin_level=10 if current_user.role == UserRole.ADMIN else 5,
            last_admin_activity=datetime.utcnow()
        )

        # Log admin access
        await user_service.track_user_activity(
            user_id=current_user.user_id,
            activity_type="admin_access",
            activity_data={
                "admin_endpoint": True,
                "admin_level": admin_user.admin_level
            }
        )

        print(f"[{datetime.utcnow()}] Admin access granted: {current_user.username}")
        return admin_user

    except HTTPException:
        raise
    except Exception as e:
        print(f"Admin authentication error: {str(e)}")
        raise AuthorizationError("Lỗi xác thực admin", "ADMIN_AUTH_ERROR")

def _get_admin_permissions(role: UserRole) -> List[str]:
    """
    Lấy danh sách permissions theo admin role
    """
    if role == UserRole.ADMIN:
        return [
            "user_management",
            "video_management",
            "system_admin",
            "emergency_controls",
            "analytics_access",
            "security_controls",
            "bulk_operations",
            "all_data_access"
        ]
    elif role == UserRole.MODERATOR:
        return [
            "content_moderation",
            "user_basic_management",
            "video_basic_management",
            "reports_access"
        ]
    else:
        return []

# ================================
# PERMISSION-BASED AUTHENTICATION
# ================================

def require_permission(permission: str):
    """
    Decorator để require specific permission
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Lấy current user từ kwargs (FastAPI sẽ inject)
            current_user = None
            for key, value in kwargs.items():
                if isinstance(value, (User, AdminUser)):
                    current_user = value
                    break

            if not current_user:
                raise AuthorizationError("Không thể xác định user hiện tại")

            # Kiểm tra permission
            user_permissions = _get_user_permissions(current_user.role)

            if permission not in user_permissions and "all_permissions" not in user_permissions:
                raise InsufficientPermissionsError(permission)

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def require_role(required_roles: List[UserRole]):
    """
    Decorator để require specific roles
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Lấy current user từ kwargs
            current_user = None
            for key, value in kwargs.items():
                if isinstance(value, (User, AdminUser)):
                    current_user = value
                    break

            if not current_user:
                raise AuthorizationError("Không thể xác định user hiện tại")

            # Kiểm tra role
            if current_user.role not in required_roles:
                raise AuthorizationError(
                    f"Cần quyền {' hoặc '.join([role.value for role in required_roles])}",
                    "INSUFFICIENT_ROLE"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def _get_user_permissions(role: UserRole) -> List[str]:
    """
    Lấy permissions theo user role
    """
    permissions_map = {
        UserRole.GUEST: ["view_public"],
        UserRole.USER: ["view_public", "upload_video", "chat", "search", "profile_edit"],
        UserRole.PREMIUM: ["view_public", "upload_video", "chat", "search", "profile_edit", "priority_processing"],
        UserRole.ENTERPRISE: ["view_public", "upload_video", "chat", "search", "profile_edit", "bulk_operations", "api_access"],
        UserRole.MODERATOR: ["view_public", "upload_video", "chat", "search", "profile_edit", "content_moderation"],
        UserRole.ADMIN: ["all_permissions"]
    }
    return permissions_map.get(role, [])

# ================================
# MIDDLEWARE CLASSES
# ================================

class AuthenticationMiddleware:
    """
    Middleware để handle authentication cho toàn bộ app
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """
        ASGI middleware call
        """
        if scope["type"] == "http":
            request = Request(scope, receive)

            # Skip authentication cho public endpoints
            if self._is_public_endpoint(request.url.path):
                await self.app(scope, receive, send)
                return

            # Thêm user info vào request state nếu có token
            try:
                auth_header = request.headers.get("authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[7:]  # Remove "Bearer "
                    user = await get_current_user_from_token(token)
                    scope["state"] = {"user": user}
            except Exception:
                # Ignore auth errors trong middleware, để endpoints tự handle
                pass

        await self.app(scope, receive, send)

    def _is_public_endpoint(self, path: str) -> bool:
        """
        Kiểm tra endpoint có phải public không
        """
        public_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/auth/login",
            "/auth/register",
            "/auth/forgot-password"
        ]

        for public_path in public_paths:
            if path.startswith(public_path):
                return True

        return False

# ================================
# SESSION MANAGEMENT
# ================================

class SessionManager:
    """
    Manager cho user sessions
    """

    @staticmethod
    async def create_session(user_id: str, token: str, device_info: Dict[str, Any] = None) -> str:
        """
        Tạo session mới cho user
        """
        try:
            session_id = f"session_{user_id}_{int(time.time())}"

            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "access_token": token,
                "created_at": time.time(),
                "last_activity": time.time(),
                "device_info": device_info or {},
                "is_active": True
            }

            # Lưu session vào cache
            await cache_service.set(
                f"session:{session_id}",
                session_data,
                ttl=auth_config.SESSION_TIMEOUT_MINUTES * 60,
                namespace='user'
            )

            # Lưu mapping user -> sessions
            user_sessions_key = f"user_sessions:{user_id}"
            user_sessions = await cache_service.get(user_sessions_key, namespace='user') or []
            user_sessions.append(session_id)

            # Giới hạn số sessions nếu cần
            if not auth_config.ALLOW_MULTIPLE_SESSIONS and len(user_sessions) > 1:
                # Invalidate session cũ
                old_session = user_sessions[0]
                await SessionManager.invalidate_session(old_session)
                user_sessions = [session_id]

            await cache_service.set(
                user_sessions_key,
                user_sessions,
                ttl=auth_config.SESSION_TIMEOUT_MINUTES * 60,
                namespace='user'
            )

            return session_id

        except Exception as e:
            print(f"Error creating session: {str(e)}")
            raise AuthenticationError("Không thể tạo session")

    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy session data
        """
        try:
            return await cache_service.get(f"session:{session_id}", namespace='user')
        except Exception as e:
            print(f"Error getting session: {str(e)}")
            return None

    @staticmethod
    async def update_session_activity(session_id: str):
        """
        Cập nhật last activity của session
        """
        try:
            session_data = await SessionManager.get_session(session_id)
            if session_data:
                session_data["last_activity"] = time.time()
                await cache_service.set(
                    f"session:{session_id}",
                    session_data,
                    ttl=auth_config.SESSION_TIMEOUT_MINUTES * 60,
                    namespace='user'
                )
        except Exception as e:
            print(f"Error updating session activity: {str(e)}")

    @staticmethod
    async def invalidate_session(session_id: str):
        """
        Invalidate session
        """
        try:
            await cache_service.delete(f"session:{session_id}", namespace='user')
        except Exception as e:
            print(f"Error invalidating session: {str(e)}")

    @staticmethod
    async def invalidate_all_user_sessions(user_id: str):
        """
        Invalidate tất cả sessions của user
        """
        try:
            user_sessions_key = f"user_sessions:{user_id}"
            user_sessions = await cache_service.get(user_sessions_key, namespace='user') or []

            for session_id in user_sessions:
                await SessionManager.invalidate_session(session_id)

            await cache_service.delete(user_sessions_key, namespace='user')

        except Exception as e:
            print(f"Error invalidating user sessions: {str(e)}")

# ================================
# RATE LIMITING
# ================================

class RateLimiter:
    """
    Rate limiter cho authentication endpoints
    """

    @staticmethod
    async def check_login_attempts(user_identifier: str) -> bool:
        """
        Kiểm tra số lần login attempts
        """
        try:
            attempts_key = f"login_attempts:{user_identifier}"
            attempts = await cache_service.get(attempts_key, namespace='user') or 0

            if attempts >= auth_config.MAX_LOGIN_ATTEMPTS:
                return False  # Account locked

            return True

        except Exception as e:
            print(f"Error checking login attempts: {str(e)}")
            return True  # Default allow nếu có lỗi

    @staticmethod
    async def record_failed_login(user_identifier: str):
        """
        Record failed login attempt
        """
        try:
            attempts_key = f"login_attempts:{user_identifier}"
            attempts = await cache_service.get(attempts_key, namespace='user') or 0
            attempts += 1

            # Cache với TTL = lockout duration
            await cache_service.set(
                attempts_key,
                attempts,
                ttl=auth_config.LOCKOUT_DURATION_MINUTES * 60,
                namespace='user'
            )

        except Exception as e:
            print(f"Error recording failed login: {str(e)}")

    @staticmethod
    async def reset_login_attempts(user_identifier: str):
        """
        Reset login attempts sau khi login thành công
        """
        try:
            attempts_key = f"login_attempts:{user_identifier}"
            await cache_service.delete(attempts_key, namespace='user')
        except Exception as e:
            print(f"Error resetting login attempts: {str(e)}")

# ================================
# UTILITY FUNCTIONS
# ================================

async def get_user_from_request(request: Request) -> Optional[User]:
    """
    Lấy user từ request state (nếu có)
    """
    try:
        return getattr(request.state, "user", None)
    except Exception:
        return None

def create_auth_header(token: str) -> Dict[str, str]:
    """
    Tạo authorization header cho API calls
    """
    return {"Authorization": f"Bearer {token}"}

async def verify_admin_session(admin_user: AdminUser) -> bool:
    """
    Verify admin session còn hợp lệ không
    """
    try:
        # Admin sessions có timeout ngắn hơn
        session_key = f"admin_session:{admin_user.user_id}"
        session_data = await cache_service.get(session_key, namespace='user')

        if not session_data:
            return False

        last_activity = session_data.get("last_activity", 0)
        current_time = time.time()

        # Kiểm tra timeout
        if current_time - last_activity > (auth_config.ADMIN_SESSION_TIMEOUT_MINUTES * 60):
            await cache_service.delete(session_key, namespace='user')
            return False

        # Update last activity
        session_data["last_activity"] = current_time
        await cache_service.set(
            session_key,
            session_data,
            ttl=auth_config.ADMIN_SESSION_TIMEOUT_MINUTES * 60,
            namespace='user'
        )

        return True

    except Exception as e:
        print(f"Error verifying admin session: {str(e)}")
        return False

# ================================
# HEALTH CHECK
# ================================

async def auth_health_check() -> Dict[str, Any]:
    """
    Health check cho authentication system
    """
    try:
        start_time = time.time()

        # Test JWT token creation/verification
        test_user_data = {
            "user_id": "test_user",
            "username": "test",
            "role": "user"
        }

        test_token = TokenManager.create_access_token(test_user_data)
        decoded_payload = TokenManager.decode_token(test_token)

        # Test cache connectivity
        cache_result = await cache_service.health_check()
        cache_ok = cache_result.get("status") == "healthy"

        # Test user service connectivity
        user_service_result = await user_service.health_check()
        user_service_ok = user_service_result.get("status") == "healthy"

        response_time = (time.time() - start_time) * 1000

        return {
            "status": "healthy" if (cache_ok and user_service_ok) else "degraded",
            "service": "auth_middleware",
            "response_time_ms": round(response_time, 2),
            "components": {
                "jwt_operations": "healthy",
                "cache_service": "healthy" if cache_ok else "error",
                "user_service": "healthy" if user_service_ok else "error"
            },
            "configuration": {
                "jwt_expire_hours": auth_config.JWT_EXPIRE_HOURS,
                "session_timeout_minutes": auth_config.SESSION_TIMEOUT_MINUTES,
                "max_login_attempts": auth_config.MAX_LOGIN_ATTEMPTS,
                "require_email_verification": auth_config.REQUIRE_EMAIL_VERIFICATION,
                "allow_multiple_sessions": auth_config.ALLOW_MULTIPLE_SESSIONS
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ================================
# EXPORTS
# ================================

__all__ = [
    # Main dependencies
    "get_current_user",
    "get_current_admin_user",

    # Token management
    "TokenManager",

    # Session management
    "SessionManager",

    # Rate limiting
    "RateLimiter",

    # Middleware
    "AuthenticationMiddleware",

    # Decorators
    "require_permission",
    "require_role",

    # Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "SessionExpiredError",
    "InsufficientPermissionsError",

    # Utilities
    "get_user_from_request",
    "create_auth_header",
    "verify_admin_session",
    "auth_health_check",

    # Configuration
    "auth_config"
]

print(f"[{datetime.utcnow()}] Authentication middleware initialized by user: xthanh1910")