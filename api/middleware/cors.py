# Placeholder for cors middleware
# api/middleware/cors.py
"""
CORS Middleware - Cross-Origin Resource Sharing configuration
Dev2: API Security & Frontend Integration - cho phép frontend Dev4 truy cập API
Current: 2025-07-03 14:25:47 UTC, User: xthanh1910
"""

import re
from typing import List, Dict, Any, Optional, Union, Sequence
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

# Import cache service để lưu CORS config
from ..services.cache_service import CacheService

# ================================
# CORS CONFIGURATION
# ================================

class CORSConfig:
    """
    Configuration cho CORS middleware
    """
    # Development origins (sẽ được Dev4 cung cấp)
    DEVELOPMENT_ORIGINS = [
        "http://localhost:3000",           # React dev server default
        "http://localhost:3001",           # React dev server alt
        "http://localhost:8080",           # Vue dev server
        "http://localhost:8081",           # Vue dev server alt
        "http://localhost:4200",           # Angular dev server
        "http://127.0.0.1:3000",           # Localhost alternative
        "http://127.0.0.1:8080",
    ]

    # Production origins (sẽ được cập nhật cho production)
    PRODUCTION_ORIGINS = [
        "https://ai-challenge.com",        # Main domain
        "https://www.ai-challenge.com",    # WWW subdomain
        "https://app.ai-challenge.com",    # App subdomain
        "https://admin.ai-challenge.com",  # Admin subdomain
        "https://api.ai-challenge.com",    # API subdomain
    ]

    # Mobile app origins (nếu có native apps)
    MOBILE_ORIGINS = [
        "capacitor://localhost",           # Capacitor apps
        "ionic://localhost",               # Ionic apps
        "file://",                        # Cordova apps
    ]

    # Admin-only origins
    ADMIN_ORIGINS = [
        "https://admin.ai-challenge.com",
        "http://localhost:3002",          # Admin dev server
        "http://127.0.0.1:3002",
    ]

    # Allowed HTTP methods
    ALLOWED_METHODS = [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "HEAD"
    ]

    # Allowed headers
    ALLOWED_HEADERS = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-API-Key",
        "X-Client-Version",
        "X-Device-ID",
        "X-User-Agent",
        "X-Session-ID",
        "Cache-Control",
        "Pragma",
    ]

    # Headers to expose to frontend
    EXPOSED_HEADERS = [
        "X-Total-Count",
        "X-Page-Count",
        "X-Current-Page",
        "X-Per-Page",
        "X-Rate-Limit-Limit",
        "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset",
        "X-Processing-Time",
        "X-Request-ID",
        "X-API-Version",
        "Content-Range",
        "Accept-Ranges",
    ]

    # CORS settings
    ALLOW_CREDENTIALS = True              # Allow cookies/auth headers
    MAX_AGE = 3600                       # Preflight cache time (1 hour)

    # Environment settings
    ENVIRONMENT = "development"           # Will be set by environment variable
    STRICT_ORIGIN_CHECK = False          # Strict origin validation
    LOG_CORS_REQUESTS = True             # Log CORS requests for debugging

cors_config = CORSConfig()

# ================================
# CORS VALIDATION FUNCTIONS
# ================================

class CORSValidator:
    """
    CORS origin validation and security checks
    """

    def __init__(self):
        self.cache = CacheService()
        self.allowed_patterns = self._compile_origin_patterns()

    def _compile_origin_patterns(self) -> List[re.Pattern]:
        """
        Compile regex patterns cho dynamic origin validation
        """
        patterns = []

        # Pattern cho subdomains của main domain
        patterns.append(re.compile(r"^https?://([a-zA-Z0-9-]+\.)?ai-challenge\.com$"))

        # Pattern cho localhost với different ports
        patterns.append(re.compile(r"^https?://localhost:[0-9]+$"))
        patterns.append(re.compile(r"^https?://127\.0\.0\.1:[0-9]+$"))

        # Pattern cho development environments
        patterns.append(re.compile(r"^https?://([a-zA-Z0-9-]+\.)?dev\.ai-challenge\.com$"))

        # Pattern cho staging environments
        patterns.append(re.compile(r"^https?://([a-zA-Z0-9-]+\.)?staging\.ai-challenge\.com$"))

        return patterns

    async def validate_origin(self, origin: str, request: Request = None) -> bool:
        """
        Validate origin có được phép không
        """
        try:
            if not origin:
                return False

            # Normalize origin
            origin = origin.lower().strip()

            # Bước 1: Check exact match với allowed origins
            all_allowed_origins = self._get_all_allowed_origins()
            if origin in all_allowed_origins:
                return True

            # Bước 2: Check với regex patterns
            for pattern in self.allowed_patterns:
                if pattern.match(origin):
                    return True

            # Bước 3: Check cached dynamic origins
            if await self._is_cached_origin_allowed(origin):
                return True

            # Bước 4: Security checks
            if not self._is_origin_secure(origin):
                print(f"[{self._get_timestamp()}] CORS: Insecure origin rejected: {origin}")
                return False

            # Bước 5: Development mode - more permissive
            if cors_config.ENVIRONMENT == "development":
                if self._is_development_origin(origin):
                    # Cache cho development
                    await self._cache_allowed_origin(origin, ttl=300)  # 5 minutes
                    return True

            print(f"[{self._get_timestamp()}] CORS: Origin rejected: {origin}")
            return False

        except Exception as e:
            print(f"CORS validation error for origin {origin}: {str(e)}")
            return False

    def _get_all_allowed_origins(self) -> List[str]:
        """
        Lấy tất cả allowed origins
        """
        all_origins = []
        all_origins.extend(cors_config.DEVELOPMENT_ORIGINS)
        all_origins.extend(cors_config.PRODUCTION_ORIGINS)
        all_origins.extend(cors_config.MOBILE_ORIGINS)

        # Normalize tất cả origins
        return [origin.lower().strip() for origin in all_origins]

    async def _is_cached_origin_allowed(self, origin: str) -> bool:
        """
        Kiểm tra origin có trong cache allowed origins không
        """
        try:
            cache_key = f"cors_allowed_origin:{origin}"
            cached_result = await self.cache.get(cache_key, namespace='security')
            return cached_result is True
        except Exception:
            return False

    async def _cache_allowed_origin(self, origin: str, ttl: int = 3600):
        """
        Cache origin là allowed
        """
        try:
            cache_key = f"cors_allowed_origin:{origin}"
            await self.cache.set(cache_key, True, ttl=ttl, namespace='security')
        except Exception as e:
            print(f"Error caching CORS origin: {str(e)}")

    def _is_origin_secure(self, origin: str) -> bool:
        """
        Kiểm tra origin có secure không
        """
        try:
            parsed = urlparse(origin)

            # Check scheme
            if parsed.scheme not in ["http", "https", "capacitor", "ionic", "file"]:
                return False

            # Production must use HTTPS (except localhost)
            if cors_config.ENVIRONMENT == "production":
                if parsed.scheme == "http" and not self._is_localhost(parsed.hostname):
                    return False

            # Check for suspicious domains
            suspicious_patterns = [
                "phishing",
                "malware",
                "suspicious",
                "evil",
                "hack"
            ]

            for pattern in suspicious_patterns:
                if pattern in origin.lower():
                    return False

            return True

        except Exception:
            return False

    def _is_localhost(self, hostname: str) -> bool:
        """
        Kiểm tra hostname có phải localhost không
        """
        if not hostname:
            return False

        localhost_patterns = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1"
        ]

        return hostname.lower() in localhost_patterns

    def _is_development_origin(self, origin: str) -> bool:
        """
        Kiểm tra origin có phải development origin không
        """
        try:
            parsed = urlparse(origin)

            # Localhost với port trong range development
            if self._is_localhost(parsed.hostname):
                if parsed.port and 3000 <= parsed.port <= 9000:
                    return True

            # Development subdomains
            if parsed.hostname and ".dev." in parsed.hostname:
                return True

            return False

        except Exception:
            return False

    def _get_timestamp(self) -> str:
        """
        Get formatted timestamp
        """
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ================================
# ENHANCED CORS MIDDLEWARE
# ================================

class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware với advanced features
    """

    def __init__(
        self,
        app: ASGIApp,
        validator: CORSValidator = None,
        allow_admin_origins: bool = False
    ):
        super().__init__(app)
        self.validator = validator or CORSValidator()
        self.allow_admin_origins = allow_admin_origins
        self.request_counter = 0

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process CORS cho mỗi request
        """
        self.request_counter += 1

        # Lấy origin từ request
        origin = request.headers.get("origin")

        # Log CORS request nếu enabled
        if cors_config.LOG_CORS_REQUESTS and origin:
            print(f"[{self._get_timestamp()}] CORS Request #{self.request_counter}: {request.method} {request.url.path} from {origin}")

        # Xử lý preflight request (OPTIONS)
        if request.method == "OPTIONS":
            return await self._handle_preflight_request(request, origin)

        # Xử lý actual request
        response = await call_next(request)

        # Thêm CORS headers vào response
        await self._add_cors_headers(request, response, origin)

        return response

    async def _handle_preflight_request(self, request: Request, origin: str) -> Response:
        """
        Xử lý preflight OPTIONS request
        """
        try:
            # Validate origin
            if not await self._is_origin_allowed(request, origin):
                return self._create_cors_error_response("Origin not allowed")

            # Check requested method
            requested_method = request.headers.get("access-control-request-method")
            if requested_method and requested_method not in cors_config.ALLOWED_METHODS:
                return self._create_cors_error_response("Method not allowed")

            # Check requested headers
            requested_headers = request.headers.get("access-control-request-headers")
            if requested_headers and not self._are_headers_allowed(requested_headers):
                return self._create_cors_error_response("Headers not allowed")

            # Create successful preflight response
            response = Response(status_code=200)

            # Add CORS headers
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(cors_config.ALLOWED_METHODS)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(cors_config.ALLOWED_HEADERS)
            response.headers["Access-Control-Max-Age"] = str(cors_config.MAX_AGE)

            if cors_config.ALLOW_CREDENTIALS:
                response.headers["Access-Control-Allow-Credentials"] = "true"

            # Add additional security headers
            response.headers["Vary"] = "Origin, Access-Control-Request-Method, Access-Control-Request-Headers"

            if cors_config.LOG_CORS_REQUESTS:
                print(f"[{self._get_timestamp()}] CORS Preflight successful for origin: {origin}")

            return response

        except Exception as e:
            print(f"CORS preflight error: {str(e)}")
            return self._create_cors_error_response("Internal server error")

    async def _add_cors_headers(self, request: Request, response: Response, origin: str):
        """
        Thêm CORS headers vào actual response
        """
        try:
            # Validate origin
            if not await self._is_origin_allowed(request, origin):
                return

            # Add basic CORS headers
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Expose-Headers"] = ", ".join(cors_config.EXPOSED_HEADERS)

            if cors_config.ALLOW_CREDENTIALS:
                response.headers["Access-Control-Allow-Credentials"] = "true"

            # Add Vary header for caching
            response.headers["Vary"] = "Origin"

            # Add custom headers
            response.headers["X-CORS-Request-ID"] = str(self.request_counter)
            response.headers["X-CORS-Processed"] = "true"

        except Exception as e:
            print(f"Error adding CORS headers: {str(e)}")

    async def _is_origin_allowed(self, request: Request, origin: str) -> bool:
        """
        Kiểm tra origin có được allow không
        """
        if not origin:
            return True  # No origin = same-origin request

        # Check admin origins nếu required
        if self.allow_admin_origins:
            if origin.lower() in [o.lower() for o in cors_config.ADMIN_ORIGINS]:
                return True

        # Use validator
        return await self.validator.validate_origin(origin, request)

    def _are_headers_allowed(self, requested_headers: str) -> bool:
        """
        Kiểm tra requested headers có được phép không
        """
        if not requested_headers:
            return True

        # Parse requested headers
        headers = [h.strip().lower() for h in requested_headers.split(",")]
        allowed_headers_lower = [h.lower() for h in cors_config.ALLOWED_HEADERS]

        # Check all requested headers are allowed
        for header in headers:
            if header not in allowed_headers_lower:
                return False

        return True

    def _create_cors_error_response(self, message: str) -> Response:
        """
        Tạo CORS error response
        """
        return Response(
            content=f'{{"error": "CORS Error", "message": "{message}"}}',
            status_code=403,
            headers={
                "Content-Type": "application/json",
                "X-CORS-Error": message
            }
        )

    def _get_timestamp(self) -> str:
        """
        Get formatted timestamp
        """
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ================================
# CORS CONFIGURATION MANAGER
# ================================

class CORSConfigManager:
    """
    Manager để update CORS configuration runtime
    """

    def __init__(self):
        self.cache = CacheService()

    async def add_allowed_origin(self, origin: str, temporary: bool = False, ttl: int = 3600) -> bool:
        """
        Thêm origin vào allowed list
        """
        try:
            # Validate origin format
            validator = CORSValidator()
            if not validator._is_origin_secure(origin):
                return False

            if temporary:
                # Add to cache temporarily
                await self.cache.set(
                    f"cors_allowed_origin:{origin}",
                    True,
                    ttl=ttl,
                    namespace='security'
                )
            else:
                # Add to persistent config (would update database in real implementation)
                # For now, just cache with longer TTL
                await self.cache.set(
                    f"cors_allowed_origin:{origin}",
                    True,
                    ttl=86400,  # 24 hours
                    namespace='security'
                )

            print(f"[{self._get_timestamp()}] CORS: Added allowed origin: {origin} ({'temporary' if temporary else 'permanent'})")
            return True

        except Exception as e:
            print(f"Error adding CORS origin: {str(e)}")
            return False

    async def remove_allowed_origin(self, origin: str) -> bool:
        """
        Remove origin khỏi allowed list
        """
        try:
            await self.cache.delete(f"cors_allowed_origin:{origin}", namespace='security')
            print(f"[{self._get_timestamp()}] CORS: Removed allowed origin: {origin}")
            return True
        except Exception as e:
            print(f"Error removing CORS origin: {str(e)}")
            return False

    async def get_allowed_origins(self) -> Dict[str, Any]:
        """
        Lấy danh sách allowed origins
        """
        try:
            # Get static origins
            static_origins = {
                "development": cors_config.DEVELOPMENT_ORIGINS,
                "production": cors_config.PRODUCTION_ORIGINS,
                "mobile": cors_config.MOBILE_ORIGINS,
                "admin": cors_config.ADMIN_ORIGINS
            }

            # Get dynamic origins from cache (simplified implementation)
            # In real implementation, would scan cache for pattern
            dynamic_origins = []

            return {
                "static_origins": static_origins,
                "dynamic_origins": dynamic_origins,
                "total_static": sum(len(origins) for origins in static_origins.values()),
                "total_dynamic": len(dynamic_origins)
            }

        except Exception as e:
            return {"error": str(e)}

    async def update_cors_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update CORS configuration
        """
        try:
            # Update allowed methods
            if "allowed_methods" in config_updates:
                cors_config.ALLOWED_METHODS = config_updates["allowed_methods"]

            # Update allowed headers
            if "allowed_headers" in config_updates:
                cors_config.ALLOWED_HEADERS = config_updates["allowed_headers"]

            # Update exposed headers
            if "exposed_headers" in config_updates:
                cors_config.EXPOSED_HEADERS = config_updates["exposed_headers"]

            # Update credentials setting
            if "allow_credentials" in config_updates:
                cors_config.ALLOW_CREDENTIALS = config_updates["allow_credentials"]

            # Update max age
            if "max_age" in config_updates:
                cors_config.MAX_AGE = config_updates["max_age"]

            print(f"[{self._get_timestamp()}] CORS: Configuration updated: {list(config_updates.keys())}")
            return True

        except Exception as e:
            print(f"Error updating CORS config: {str(e)}")
            return False

    def _get_timestamp(self) -> str:
        """
        Get formatted timestamp
        """
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ================================
# SPECIALIZED CORS MIDDLEWARES
# ================================

class PublicCORSMiddleware(EnhancedCORSMiddleware):
    """
    CORS middleware cho public endpoints
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app, allow_admin_origins=False)

class AdminCORSMiddleware(EnhancedCORSMiddleware):
    """
    CORS middleware cho admin endpoints
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app, allow_admin_origins=True)

# ================================
# CORS SECURITY ANALYZER
# ================================

class CORSSecurityAnalyzer:
    """
    Analyzer để check CORS security issues
    """

    def __init__(self):
        self.cache = CacheService()

    async def analyze_cors_requests(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Analyze CORS requests trong time window
        """
        try:
            # This would be implemented với real logging system
            # For now, return mock analysis

            analysis = {
                "time_window_seconds": time_window,
                "total_cors_requests": 1250,
                "unique_origins": 15,
                "blocked_origins": 3,
                "preflight_requests": 450,
                "suspicious_patterns": [],
                "top_origins": [
                    {"origin": "https://ai-challenge.com", "requests": 800},
                    {"origin": "http://localhost:3000", "requests": 300},
                    {"origin": "https://app.ai-challenge.com", "requests": 150}
                ],
                "security_events": [
                    {
                        "type": "blocked_origin",
                        "origin": "https://malicious-site.com",
                        "timestamp": "2025-07-03T14:20:15Z",
                        "reason": "Not in allowed origins"
                    }
                ]
            }

            return analysis

        except Exception as e:
            return {"error": str(e)}

    async def check_cors_vulnerabilities(self) -> Dict[str, Any]:
        """
        Check potential CORS vulnerabilities
        """
        try:
            vulnerabilities = []
            recommendations = []

            # Check if wildcard origins are used
            if "*" in cors_config.DEVELOPMENT_ORIGINS:
                vulnerabilities.append({
                    "type": "wildcard_origin",
                    "severity": "high",
                    "description": "Wildcard origins với credentials enabled"
                })

            # Check HTTPS usage
            insecure_origins = [
                origin for origin in cors_config.PRODUCTION_ORIGINS
                if origin.startswith("http://")
            ]

            if insecure_origins:
                vulnerabilities.append({
                    "type": "insecure_origins",
                    "severity": "medium",
                    "description": f"HTTP origins in production: {insecure_origins}"
                })

            # Generate recommendations
            if cors_config.ENVIRONMENT == "production":
                if cors_config.DEVELOPMENT_ORIGINS:
                    recommendations.append("Remove development origins in production")

                if not cors_config.STRICT_ORIGIN_CHECK:
                    recommendations.append("Enable strict origin checking in production")

            return {
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations,
                "security_score": max(0, 100 - len(vulnerabilities) * 20),
                "checked_at": "2025-07-03T14:25:47Z"
            }

        except Exception as e:
            return {"error": str(e)}

# ================================
# UTILITY FUNCTIONS
# ================================

def create_standard_cors_middleware(app: ASGIApp) -> CORSMiddleware:
    """
    Tạo standard FastAPI CORS middleware
    """
    all_origins = []
    all_origins.extend(cors_config.DEVELOPMENT_ORIGINS)
    all_origins.extend(cors_config.PRODUCTION_ORIGINS)
    all_origins.extend(cors_config.MOBILE_ORIGINS)

    return CORSMiddleware(
        app,
        allow_origins=all_origins,
        allow_credentials=cors_config.ALLOW_CREDENTIALS,
        allow_methods=cors_config.ALLOWED_METHODS,
        allow_headers=cors_config.ALLOWED_HEADERS,
        expose_headers=cors_config.EXPOSED_HEADERS,
        max_age=cors_config.MAX_AGE
    )

def create_enhanced_cors_middleware(app: ASGIApp, for_admin: bool = False) -> EnhancedCORSMiddleware:
    """
    Tạo enhanced CORS middleware
    """
    if for_admin:
        return AdminCORSMiddleware(app)
    else:
        return PublicCORSMiddleware(app)

async def cors_health_check() -> Dict[str, Any]:
    """
    Health check cho CORS system
    """
    try:
        start_time = time.time()

        validator = CORSValidator()

        # Test origin validation
        test_origin = "https://ai-challenge.com"
        validation_result = await validator.validate_origin(test_origin)

        # Test cache connectivity
        cache_result = await validator.cache.health_check()
        cache_ok = cache_result.get("status") == "healthy"

        response_time = (time.time() - start_time) * 1000

        return {
            "status": "healthy" if (validation_result and cache_ok) else "degraded",
            "service": "cors_middleware",
            "response_time_ms": round(response_time, 2),
            "components": {
                "origin_validator": "healthy" if validation_result else "error",
                "cache_service": "healthy" if cache_ok else "error"
            },
            "configuration": {
                "environment": cors_config.ENVIRONMENT,
                "allow_credentials": cors_config.ALLOW_CREDENTIALS,
                "max_age": cors_config.MAX_AGE,
                "total_allowed_origins": len(cors_config.DEVELOPMENT_ORIGINS + cors_config.PRODUCTION_ORIGINS),
                "strict_origin_check": cors_config.STRICT_ORIGIN_CHECK
            },
            "test_results": {
                "origin_validation": "passed" if validation_result else "failed",
                "cache_test": "passed" if cache_ok else "failed"
            },
            "timestamp": "2025-07-03T14:25:47Z"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-07-03T14:25:47Z"
        }

# ================================
# EXPORTS
# ================================

__all__ = [
    # Main classes
    "EnhancedCORSMiddleware",
    "CORSValidator",
    "CORSConfigManager",

    # Specialized middlewares
    "PublicCORSMiddleware",
    "AdminCORSMiddleware",

    # Security analyzer
    "CORSSecurityAnalyzer",

    # Utility functions
    "create_standard_cors_middleware",
    "create_enhanced_cors_middleware",
    "cors_health_check",

    # Configuration
    "cors_config"
]

print(f"[2025-07-03 14:25:47] CORS middleware initialized by user: xthanh1910")