# Placeholder for rate limit middleware
# api/middleware/rate_limit.py
"""
Rate Limiting Middleware - Quản lý rate limiting cho API endpoints
Dev2: API Security & Performance - bảo vệ API khỏi spam và abuse
Current: 2025-07-03 14:23:28 UTC, User: xthanh1910
"""

import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from collections import defaultdict

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import cache service để lưu rate limiting data
from ..services.cache_service import CacheService
from ..models.user_models import UserRole

# ================================
# RATE LIMITING CONFIGURATION
# ================================

class RateLimitConfig:
    """
    Configuration cho rate limiting system
    """
    # Default rate limits (requests per minute)
    DEFAULT_RATE_LIMIT = 60        # 60 requests/minute cho anonymous
    USER_RATE_LIMIT = 120          # 120 requests/minute cho users
    PREMIUM_RATE_LIMIT = 300       # 300 requests/minute cho premium users
    ADMIN_RATE_LIMIT = 1000        # 1000 requests/minute cho admins

    # Sliding window settings
    WINDOW_SIZE_SECONDS = 60       # 1 minute window
    BUCKET_SIZE_SECONDS = 5        # 5 second buckets for sliding window

    # Burst settings
    ENABLE_BURST_MODE = True       # Cho phép burst requests
    BURST_MULTIPLIER = 2           # Burst = rate_limit * multiplier
    BURST_WINDOW_SECONDS = 10      # Burst window

    # Specific endpoint limits
    ENDPOINT_LIMITS = {
        # Authentication endpoints (more restrictive)
        "POST /auth/login": 10,           # 10 login attempts per minute
        "POST /auth/register": 5,         # 5 registrations per minute
        "POST /auth/forgot-password": 3,  # 3 password resets per minute

        # Video upload endpoints
        "POST /videos/upload": 10,        # 10 uploads per minute
        "POST /videos/batch-upload": 2,   # 2 batch uploads per minute

        # Chat endpoints
        "POST /chat/message": 30,         # 30 chat messages per minute
        "POST /chat/quick": 60,           # 60 quick chats per minute

        # Search endpoints
        "GET /search": 100,               # 100 searches per minute
        "GET /videos/search": 100,        # 100 video searches per minute

        # Admin endpoints (less restrictive)
        "GET /admin/dashboard": 30,       # 30 dashboard loads per minute
        "POST /admin/emergency/*": 10,    # 10 emergency actions per minute
    }

    # IP-based global limits
    IP_GLOBAL_LIMIT = 200             # 200 requests/minute per IP
    IP_BLACKLIST_THRESHOLD = 1000     # Ban IP if exceeds 1000 requests/minute
    IP_BLACKLIST_DURATION = 3600      # Ban for 1 hour

    # Cache settings
    CACHE_TTL = 3600                  # 1 hour cache for rate limit data
    CLEANUP_INTERVAL = 300            # Cleanup old data every 5 minutes

rate_limit_config = RateLimitConfig()

# ================================
# EXCEPTION CLASSES
# ================================

class RateLimitExceeded(HTTPException):
    """
    Rate limit exceeded exception
    """
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit: int = 0,
        remaining: int = 0,
        reset_time: int = 0
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
                "X-Error-Code": "RATE_LIMIT_EXCEEDED"
            }
        )

class IPBlacklistedException(HTTPException):
    """
    IP blacklisted exception
    """
    def __init__(self, detail: str = "IP address has been temporarily blacklisted"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers={"X-Error-Code": "IP_BLACKLISTED"}
        )

# ================================
# RATE LIMITER CLASSES
# ================================

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation
    """

    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.cleanup_task = None

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int = None
    ) -> tuple[bool, Dict[str, int]]:
        """
        Kiểm tra request có được phép không (sliding window algorithm)
        Returns: (allowed, info_dict)
        """
        try:
            window_seconds = window_seconds or rate_limit_config.WINDOW_SIZE_SECONDS
            bucket_size = rate_limit_config.BUCKET_SIZE_SECONDS
            current_time = int(time.time())

            # Tính số buckets trong window
            num_buckets = window_seconds // bucket_size

            # Lấy data từ cache
            cache_key = f"rate_limit:{key}"
            rate_data = await self.cache.get(cache_key, namespace='rate_limit') or {}

            # Clean up old buckets
            cutoff_time = current_time - window_seconds
            rate_data = {
                timestamp: count
                for timestamp, count in rate_data.items()
                if int(timestamp) > cutoff_time
            }

            # Tính current bucket
            current_bucket = (current_time // bucket_size) * bucket_size

            # Tính tổng requests trong window
            total_requests = sum(rate_data.values())

            # Kiểm tra có vượt limit không
            if total_requests >= limit:
                # Tính thời gian reset (oldest bucket + window)
                oldest_bucket = min(map(int, rate_data.keys())) if rate_data else current_bucket
                reset_time = oldest_bucket + window_seconds

                info = {
                    "limit": limit,
                    "remaining": 0,
                    "reset_time": reset_time,
                    "retry_after": max(1, reset_time - current_time)
                }

                return False, info

            # Add current request
            rate_data[str(current_bucket)] = rate_data.get(str(current_bucket), 0) + 1

            # Save back to cache
            await self.cache.set(
                cache_key,
                rate_data,
                ttl=window_seconds + 60,  # Extra TTL for safety
                namespace='rate_limit'
            )

            # Calculate remaining requests
            remaining = max(0, limit - sum(rate_data.values()))

            info = {
                "limit": limit,
                "remaining": remaining,
                "reset_time": current_bucket + window_seconds,
                "retry_after": 0
            }

            return True, info

        except Exception as e:
            print(f"Rate limiter error: {str(e)}")
            # Fallback: allow request on error
            return True, {"limit": limit, "remaining": limit, "reset_time": 0, "retry_after": 0}

    async def reset_limit(self, key: str):
        """
        Reset rate limit cho một key
        """
        try:
            cache_key = f"rate_limit:{key}"
            await self.cache.delete(cache_key, namespace='rate_limit')
        except Exception as e:
            print(f"Error resetting rate limit: {str(e)}")

    async def get_usage_stats(self, key: str, window_seconds: int = None) -> Dict[str, Any]:
        """
        Lấy thống kê usage cho key
        """
        try:
            window_seconds = window_seconds or rate_limit_config.WINDOW_SIZE_SECONDS
            cache_key = f"rate_limit:{key}"
            rate_data = await self.cache.get(cache_key, namespace='rate_limit') or {}

            current_time = int(time.time())
            cutoff_time = current_time - window_seconds

            # Filter data trong window
            filtered_data = {
                timestamp: count
                for timestamp, count in rate_data.items()
                if int(timestamp) > cutoff_time
            }

            total_requests = sum(filtered_data.values())
            bucket_count = len(filtered_data)
            avg_per_bucket = total_requests / max(bucket_count, 1)

            return {
                "total_requests": total_requests,
                "bucket_count": bucket_count,
                "avg_requests_per_bucket": round(avg_per_bucket, 2),
                "window_seconds": window_seconds,
                "data_points": filtered_data
            }

        except Exception as e:
            print(f"Error getting usage stats: {str(e)}")
            return {}

class BurstRateLimiter:
    """
    Burst rate limiter cho short-term spikes
    """

    def __init__(self, cache_service: CacheService):
        self.cache = cache_service

    async def is_burst_allowed(
        self,
        key: str,
        base_limit: int,
        burst_multiplier: float = None
    ) -> tuple[bool, Dict[str, int]]:
        """
        Kiểm tra burst requests
        """
        try:
            burst_multiplier = burst_multiplier or rate_limit_config.BURST_MULTIPLIER
            burst_limit = int(base_limit * burst_multiplier)
            burst_window = rate_limit_config.BURST_WINDOW_SECONDS

            return await SlidingWindowRateLimiter(self.cache).is_allowed(
                f"burst:{key}",
                burst_limit,
                burst_window
            )

        except Exception as e:
            print(f"Burst rate limiter error: {str(e)}")
            return True, {"limit": burst_limit, "remaining": burst_limit, "reset_time": 0, "retry_after": 0}

# ================================
# IP BLACKLIST MANAGER
# ================================

class IPBlacklistManager:
    """
    Quản lý IP blacklist
    """

    def __init__(self, cache_service: CacheService):
        self.cache = cache_service

    async def is_blacklisted(self, ip_address: str) -> bool:
        """
        Kiểm tra IP có bị blacklist không
        """
        try:
            blacklist_key = f"ip_blacklist:{ip_address}"
            blacklist_data = await self.cache.get(blacklist_key, namespace='security')

            if blacklist_data:
                # Kiểm tra có hết hạn chưa
                blacklist_until = blacklist_data.get("until", 0)
                if time.time() < blacklist_until:
                    return True
                else:
                    # Hết hạn blacklist, xóa khỏi cache
                    await self.cache.delete(blacklist_key, namespace='security')

            return False

        except Exception as e:
            print(f"Error checking IP blacklist: {str(e)}")
            return False  # Default allow on error

    async def add_to_blacklist(
        self,
        ip_address: str,
        duration_seconds: int = None,
        reason: str = "Rate limit violation"
    ):
        """
        Thêm IP vào blacklist
        """
        try:
            duration_seconds = duration_seconds or rate_limit_config.IP_BLACKLIST_DURATION
            blacklist_until = time.time() + duration_seconds

            blacklist_data = {
                "ip": ip_address,
                "blacklisted_at": time.time(),
                "until": blacklist_until,
                "reason": reason,
                "duration_seconds": duration_seconds
            }

            blacklist_key = f"ip_blacklist:{ip_address}"
            await self.cache.set(
                blacklist_key,
                blacklist_data,
                ttl=duration_seconds + 60,  # Extra TTL
                namespace='security'
            )

            # Log blacklist action
            print(f"[{datetime.utcnow()}] IP blacklisted: {ip_address} for {duration_seconds}s, reason: {reason}")

        except Exception as e:
            print(f"Error adding IP to blacklist: {str(e)}")

    async def remove_from_blacklist(self, ip_address: str):
        """
        Xóa IP khỏi blacklist
        """
        try:
            blacklist_key = f"ip_blacklist:{ip_address}"
            await self.cache.delete(blacklist_key, namespace='security')
            print(f"[{datetime.utcnow()}] IP removed from blacklist: {ip_address}")
        except Exception as e:
            print(f"Error removing IP from blacklist: {str(e)}")

    async def get_blacklist_info(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin blacklist của IP
        """
        try:
            blacklist_key = f"ip_blacklist:{ip_address}"
            return await self.cache.get(blacklist_key, namespace='security')
        except Exception as e:
            print(f"Error getting blacklist info: {str(e)}")
            return None

# ================================
# RATE LIMITING FUNCTIONS
# ================================

class RateLimitManager:
    """
    Main rate limit manager
    """

    def __init__(self):
        self.cache = CacheService()
        self.sliding_limiter = SlidingWindowRateLimiter(self.cache)
        self.burst_limiter = BurstRateLimiter(self.cache)
        self.ip_blacklist = IPBlacklistManager(self.cache)

    async def check_rate_limit(
        self,
        request: Request,
        user_id: str = None,
        user_role: UserRole = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Main rate limit check function
        """
        try:
            # Lấy client IP
            client_ip = self._get_client_ip(request)

            # Bước 1: Kiểm tra IP blacklist
            if await self.ip_blacklist.is_blacklisted(client_ip):
                raise IPBlacklistedException()

            # Bước 2: Tạo rate limit key
            endpoint = f"{request.method} {request.url.path}"
            rate_key = self._generate_rate_key(client_ip, user_id, endpoint)

            # Bước 3: Xác định rate limit
            rate_limit = self._get_rate_limit(endpoint, user_role)

            # Bước 4: Kiểm tra sliding window rate limit
            allowed, info = await self.sliding_limiter.is_allowed(rate_key, rate_limit)

            if not allowed:
                # Bước 5: Kiểm tra burst limit nếu regular limit bị vượt
                if rate_limit_config.ENABLE_BURST_MODE:
                    burst_allowed, burst_info = await self.burst_limiter.is_burst_allowed(
                        rate_key, rate_limit
                    )

                    if burst_allowed:
                        info.update({"burst_mode": True})
                        return True, info

                # Bước 6: Kiểm tra IP global limit và blacklist nếu cần
                await self._check_and_blacklist_ip(client_ip)

                return False, info

            return True, info

        except HTTPException:
            raise
        except Exception as e:
            print(f"Rate limit check error: {str(e)}")
            # Default allow on system error
            return True, {"limit": 0, "remaining": 0, "reset_time": 0, "retry_after": 0}

    def _get_client_ip(self, request: Request) -> str:
        """
        Lấy client IP từ request (handle proxy headers)
        """
        # Check X-Forwarded-For header (from proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"

    def _generate_rate_key(self, client_ip: str, user_id: str, endpoint: str) -> str:
        """
        Tạo unique key cho rate limiting
        """
        if user_id:
            # User-based rate limiting
            base_key = f"user:{user_id}"
        else:
            # IP-based rate limiting
            base_key = f"ip:{client_ip}"

        # Include endpoint for endpoint-specific limits
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        return f"{base_key}:endpoint:{endpoint_hash}"

    def _get_rate_limit(self, endpoint: str, user_role: UserRole = None) -> int:
        """
        Xác định rate limit cho endpoint và user role
        """
        # Bước 1: Kiểm tra endpoint-specific limits
        for pattern, limit in rate_limit_config.ENDPOINT_LIMITS.items():
            if self._match_endpoint_pattern(endpoint, pattern):
                return limit

        # Bước 2: Rate limit theo user role
        if user_role == UserRole.ADMIN:
            return rate_limit_config.ADMIN_RATE_LIMIT
        elif user_role == UserRole.PREMIUM:
            return rate_limit_config.PREMIUM_RATE_LIMIT
        elif user_role in [UserRole.USER, UserRole.MODERATOR]:
            return rate_limit_config.USER_RATE_LIMIT
        else:
            return rate_limit_config.DEFAULT_RATE_LIMIT

    def _match_endpoint_pattern(self, endpoint: str, pattern: str) -> bool:
        """
        Match endpoint với pattern (support wildcards)
        """
        if "*" not in pattern:
            return endpoint == pattern

        # Simple wildcard matching
        parts = pattern.split("*")
        if len(parts) == 2:
            prefix, suffix = parts
            return endpoint.startswith(prefix) and endpoint.endswith(suffix)

        return False

    async def _check_and_blacklist_ip(self, client_ip: str):
        """
        Kiểm tra và blacklist IP nếu vượt global limit
        """
        try:
            ip_key = f"ip_global:{client_ip}"
            allowed, info = await self.sliding_limiter.is_allowed(
                ip_key,
                rate_limit_config.IP_GLOBAL_LIMIT
            )

            if not allowed:
                # Get total requests để check blacklist threshold
                usage_stats = await self.sliding_limiter.get_usage_stats(ip_key)
                total_requests = usage_stats.get("total_requests", 0)

                if total_requests >= rate_limit_config.IP_BLACKLIST_THRESHOLD:
                    await self.ip_blacklist.add_to_blacklist(
                        client_ip,
                        reason=f"Exceeded global IP limit: {total_requests} requests"
                    )

        except Exception as e:
            print(f"Error checking IP global limit: {str(e)}")

# ================================
# MIDDLEWARE IMPLEMENTATION
# ================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware cho rate limiting
    """

    def __init__(self, app, rate_manager: RateLimitManager = None):
        super().__init__(app)
        self.rate_manager = rate_manager or RateLimitManager()
        self.excluded_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/favicon.ico"
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request với rate limiting
        """
        try:
            # Skip rate limiting cho excluded paths
            if self._is_excluded_path(request.url.path):
                return await call_next(request)

            # Lấy user info từ request state (nếu có từ auth middleware)
            user_id = None
            user_role = None

            if hasattr(request.state, "user"):
                user = request.state.user
                user_id = getattr(user, "user_id", None)
                user_role = getattr(user, "role", None)

            # Kiểm tra rate limit
            allowed, rate_info = await self.rate_manager.check_rate_limit(
                request, user_id, user_role
            )

            if not allowed:
                # Tạo rate limit error response
                error_detail = "Rate limit exceeded. Too many requests."
                if rate_info.get("retry_after"):
                    error_detail += f" Try again in {rate_info['retry_after']} seconds."

                raise RateLimitExceeded(
                    detail=error_detail,
                    retry_after=rate_info.get("retry_after", 60),
                    limit=rate_info.get("limit", 0),
                    remaining=rate_info.get("remaining", 0),
                    reset_time=rate_info.get("reset_time", 0)
                )

            # Process request
            response = await call_next(request)

            # Thêm rate limit headers vào response
            self._add_rate_limit_headers(response, rate_info)

            return response

        except HTTPException as e:
            # Trả về error response với proper headers
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail, "error_code": "RATE_LIMIT_ERROR"},
                headers=e.headers
            )
        except Exception as e:
            print(f"Rate limit middleware error: {str(e)}")
            # Continue on system error
            return await call_next(request)

    def _is_excluded_path(self, path: str) -> bool:
        """
        Kiểm tra path có được exclude khỏi rate limiting không
        """
        for excluded_path in self.excluded_paths:
            if path.startswith(excluded_path):
                return True
        return False

    def _add_rate_limit_headers(self, response: Response, rate_info: Dict[str, Any]):
        """
        Thêm rate limit headers vào response
        """
        try:
            response.headers["X-RateLimit-Limit"] = str(rate_info.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset_time", 0))

            if rate_info.get("burst_mode"):
                response.headers["X-RateLimit-Burst"] = "true"

        except Exception as e:
            print(f"Error adding rate limit headers: {str(e)}")

# ================================
# DECORATORS
# ================================

def rate_limit(limit: int, window_seconds: int = 60):
    """
    Decorator để apply custom rate limit cho specific endpoints
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Fallback: proceed without rate limiting
                return await func(*args, **kwargs)

            # Apply custom rate limit
            rate_manager = RateLimitManager()

            # Generate custom key for this specific endpoint
            client_ip = rate_manager._get_client_ip(request)
            endpoint = f"{request.method} {request.url.path}"
            custom_key = f"custom:{client_ip}:{hashlib.md5(endpoint.encode()).hexdigest()[:8]}"

            allowed, info = await rate_manager.sliding_limiter.is_allowed(
                custom_key, limit, window_seconds
            )

            if not allowed:
                raise RateLimitExceeded(
                    detail=f"Custom rate limit exceeded: {limit} requests per {window_seconds} seconds",
                    retry_after=info.get("retry_after", 60),
                    limit=info.get("limit", limit),
                    remaining=info.get("remaining", 0),
                    reset_time=info.get("reset_time", 0)
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

def exempt_from_rate_limit(func: Callable) -> Callable:
    """
    Decorator để exempt endpoint khỏi rate limiting
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    # Mark function as exempt
    wrapper._rate_limit_exempt = True
    return wrapper

# ================================
# ADMIN CONTROLS
# ================================

class RateLimitAdmin:
    """
    Admin controls cho rate limiting system
    """

    def __init__(self, rate_manager: RateLimitManager):
        self.rate_manager = rate_manager

    async def get_rate_limit_stats(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Lấy thống kê rate limiting
        """
        try:
            # Implementation sẽ phức tạp hơn, đây là basic version
            cache_stats = await self.rate_manager.cache.get_cache_stats()

            return {
                "cache_stats": cache_stats,
                "config": {
                    "default_rate_limit": rate_limit_config.DEFAULT_RATE_LIMIT,
                    "user_rate_limit": rate_limit_config.USER_RATE_LIMIT,
                    "admin_rate_limit": rate_limit_config.ADMIN_RATE_LIMIT,
                    "window_size_seconds": rate_limit_config.WINDOW_SIZE_SECONDS,
                    "burst_enabled": rate_limit_config.ENABLE_BURST_MODE
                },
                "blacklist_stats": {
                    "blacklist_threshold": rate_limit_config.IP_BLACKLIST_THRESHOLD,
                    "blacklist_duration": rate_limit_config.IP_BLACKLIST_DURATION
                }
            }

        except Exception as e:
            return {"error": str(e)}

    async def reset_user_rate_limit(self, user_id: str) -> bool:
        """
        Reset rate limit cho user
        """
        try:
            # Reset tất cả keys liên quan đến user
            pattern = f"rate_limit:user:{user_id}:*"
            deleted_count = await self.rate_manager.cache.delete_pattern(
                pattern, namespace='rate_limit'
            )

            print(f"[{datetime.utcnow()}] Reset rate limit for user {user_id}: {deleted_count} keys deleted")
            return True

        except Exception as e:
            print(f"Error resetting user rate limit: {str(e)}")
            return False

    async def reset_ip_rate_limit(self, ip_address: str) -> bool:
        """
        Reset rate limit cho IP
        """
        try:
            # Reset tất cả keys liên quan đến IP
            pattern = f"rate_limit:ip:{ip_address}:*"
            deleted_count = await self.rate_manager.cache.delete_pattern(
                pattern, namespace='rate_limit'
            )

            # Remove from blacklist nếu có
            await self.rate_manager.ip_blacklist.remove_from_blacklist(ip_address)

            print(f"[{datetime.utcnow()}] Reset rate limit for IP {ip_address}: {deleted_count} keys deleted")
            return True

        except Exception as e:
            print(f"Error resetting IP rate limit: {str(e)}")
            return False

    async def emergency_clear_all_limits(self) -> Dict[str, Any]:
        """
        Emergency clear tất cả rate limits (admin only)
        """
        try:
            # Clear tất cả rate limiting data
            pattern = "rate_limit:*"
            deleted_count = await self.rate_manager.cache.delete_pattern(
                pattern, namespace='rate_limit'
            )

            # Clear IP blacklist
            blacklist_pattern = "ip_blacklist:*"
            blacklist_deleted = await self.rate_manager.cache.delete_pattern(
                blacklist_pattern, namespace='security'
            )

            result = {
                "success": True,
                "rate_limits_cleared": deleted_count,
                "blacklist_cleared": blacklist_deleted,
                "cleared_at": datetime.utcnow().isoformat(),
                "cleared_by": "emergency_action"
            }

            print(f"[{datetime.utcnow()}] Emergency rate limit clear: {result}")
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ================================
# HEALTH CHECK
# ================================

async def rate_limit_health_check() -> Dict[str, Any]:
    """
    Health check cho rate limiting system
    """
    try:
        start_time = time.time()

        rate_manager = RateLimitManager()

        # Test basic rate limiting
        test_key = f"health_check_{int(time.time())}"
        allowed, info = await rate_manager.sliding_limiter.is_allowed(test_key, 100)

        # Test cache connectivity
        cache_result = await rate_manager.cache.health_check()
        cache_ok = cache_result.get("status") == "healthy"

        # Test IP blacklist
        test_ip = "127.0.0.1"
        blacklist_ok = not await rate_manager.ip_blacklist.is_blacklisted(test_ip)

        response_time = (time.time() - start_time) * 1000

        return {
            "status": "healthy" if (cache_ok and blacklist_ok) else "degraded",
            "service": "rate_limit_middleware",
            "response_time_ms": round(response_time, 2),
            "components": {
                "sliding_window_limiter": "healthy" if allowed else "error",
                "cache_service": "healthy" if cache_ok else "error",
                "ip_blacklist": "healthy" if blacklist_ok else "error"
            },
            "configuration": {
                "default_rate_limit": rate_limit_config.DEFAULT_RATE_LIMIT,
                "window_size_seconds": rate_limit_config.WINDOW_SIZE_SECONDS,
                "burst_enabled": rate_limit_config.ENABLE_BURST_MODE,
                "ip_blacklist_enabled": True
            },
            "test_results": {
                "rate_limit_test": "passed" if allowed else "failed",
                "cache_test": "passed" if cache_ok else "failed",
                "blacklist_test": "passed" if blacklist_ok else "failed"
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
    # Main classes
    "RateLimitManager",
    "RateLimitMiddleware",
    "RateLimitAdmin",

    # Rate limiter implementations
    "SlidingWindowRateLimiter",
    "BurstRateLimiter",
    "IPBlacklistManager",

    # Exceptions
    "RateLimitExceeded",
    "IPBlacklistedException",

    # Decorators
    "rate_limit",
    "exempt_from_rate_limit",

    # Utilities
    "rate_limit_health_check",

    # Configuration
    "rate_limit_config"
]

print(f"[{datetime.utcnow()}] Rate limiting middleware initialized by user: xthanh1910")