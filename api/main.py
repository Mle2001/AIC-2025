# api/main.py
"""
Main FastAPI Application - Entry point cho AI Challenge API System
Dev2: API Server & Integration - khởi tạo và cấu hình toàn bộ API server
Current: 2025-07-03 14:33:04 UTC, User: xthanh1910
"""

import os
import asyncio
import uvloop
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

import uvicorn
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

# Import all services
from .services.user_service import user_service
from .services.video_service import video_service
from .services.chat_service import chat_service
from .services.cache_service import cache_service

# Import middleware
from .middleware.auth import AuthenticationMiddleware, get_current_user
from .middleware.rate_limit import RateLimitMiddleware, RateLimitManager
from .middleware.cors import create_enhanced_cors_middleware

# Import routers
from .routes.auth import router as auth_router
from .routes.users import router as users_router
from .routes.videos import router as videos_router
from .routes.chat import router as chat_router
from .routes.admin import router as admin_router
from .routes.health import router as health_router

# Import agents manager
from .agents_manager import agents_manager

#======================================================================================================================================
# APPLICATION LIFESPAN MANAGEMENT
#======================================================================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management - startup và shutdown logic
    """
    startup_time = datetime.utcnow()
    print(f"[{startup_time}] 🚀 Starting AI Challenge API Server...")
    print(f"[{startup_time}] 👤 Started by user: xthanh1910")

    try:
        #=================================================================================================================================
        # STARTUP SEQUENCE
        #=================================================================================================================================

        print(f"[{datetime.utcnow()}] 📋 Starting initialization sequence...")

        # Bước 1: Initialize cache service (first - other services depend on it)
        print(f"[{datetime.utcnow()}] 🗄️  Initializing cache service...")
        await cache_service.initialize()

        # Bước 2: Initialize agents manager (Dev1 integration)
        print(f"[{datetime.utcnow()}] 🤖 Initializing AI agents manager...")
        await agents_manager.initialize()

        # Bước 3: Initialize user service
        print(f"[{datetime.utcnow()}] 👥 Initializing user service...")
        await user_service.initialize()

        # Bước 4: Initialize video service
        print(f"[{datetime.utcnow()}] 🎥 Initializing video service...")
        await video_service.initialize()

        # Bước 5: Initialize chat service
        print(f"[{datetime.utcnow()}] 💬 Initializing chat service...")
        await chat_service.initialize()

        # Bước 6: Create necessary directories
        print(f"[{datetime.utcnow()}] 📁 Creating application directories...")
        await _create_app_directories()

        # Bước 7: Run startup health checks
        print(f"[{datetime.utcnow()}] 🔍 Running startup health checks...")
        health_status = await _run_startup_health_checks()

        if health_status['overall_status'] != 'healthy':
            print(f"[{datetime.utcnow()}] ⚠️  Warning: Some services are not healthy")
            for service, status in health_status['services'].items():
                if status != 'healthy':
                    print(f"[{datetime.utcnow()}] ❌ {service}: {status}")

        # Bước 8: Log successful startup
        startup_duration = (datetime.utcnow() - startup_time).total_seconds()
        print(f"[{datetime.utcnow()}] ✅ AI Challenge API started successfully!")
        print(f"[{datetime.utcnow()}] ⏱️  Startup completed in {startup_duration:.2f} seconds")
        print(f"[{datetime.utcnow()}] 🌐 Server ready to accept connections")
        print(f"[{datetime.utcnow()}] 📚 API Documentation: http://localhost:8000/docs")
        print(f"[{datetime.utcnow()}] 🔧 Admin Panel: http://localhost:8000/admin")

        # Store startup info in app state
        app.state.startup_time = startup_time
        app.state.health_status = health_status

        yield  # Server is running

        #=================================================================================================================================
        # SHUTDOWN SEQUENCE
        #=================================================================================================================================

        shutdown_time = datetime.utcnow()
        print(f"[{shutdown_time}] 🛑 Shutting down AI Challenge API Server...")

        # Graceful shutdown của tất cả services
        print(f"[{datetime.utcnow()}] 💬 Shutting down chat service...")
        await chat_service.shutdown()

        print(f"[{datetime.utcnow()}] 🎥 Shutting down video service...")
        await video_service.shutdown()

        print(f"[{datetime.utcnow()}] 👥 Shutting down user service...")
        await user_service.shutdown()

        print(f"[{datetime.utcnow()}] 🤖 Shutting down agents manager...")
        await agents_manager.shutdown()

        print(f"[{datetime.utcnow()}] 🗄️  Shutting down cache service...")
        await cache_service.close()

        shutdown_duration = (datetime.utcnow() - shutdown_time).total_seconds()
        print(f"[{datetime.utcnow()}] ✅ Graceful shutdown completed in {shutdown_duration:.2f} seconds")
        print(f"[{datetime.utcnow()}] 👋 AI Challenge API Server stopped")

    except Exception as e:
        print(f"[{datetime.utcnow()}] ❌ Error during application lifecycle: {str(e)}")
        raise

async def _create_app_directories():
    """
    Tạo các thư mục cần thiết cho application
    """
    try:
        directories = [
            "uploads",
            "uploads/videos",
            "uploads/temp",
            "uploads/processed",
            "uploads/thumbnails",
            "logs",
            "cache",
            "static",
            "static/admin"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"[{datetime.utcnow()}] ✅ Application directories created")

    except Exception as e:
        print(f"[{datetime.utcnow()}] ❌ Error creating directories: {str(e)}")

async def _run_startup_health_checks() -> Dict[str, Any]:
    """
    Chạy health checks cho tất cả services khi startup
    """
    try:
        health_results = {}

        # Check cache service
        cache_health = await cache_service.health_check()
        health_results['cache_service'] = cache_health.get('status', 'unknown')

        # Check agents manager
        agents_health = await agents_manager.health_check()
        health_results['agents_manager'] = agents_health.get('status', 'unknown')

        # Check user service
        user_health = await user_service.health_check()
        health_results['user_service'] = user_health.get('status', 'unknown')

        # Check video service
        video_health = await video_service.health_check()
        health_results['video_service'] = video_health.get('status', 'unknown')

        # Check chat service
        chat_health = await chat_service.health_check()
        health_results['chat_service'] = chat_health.get('status', 'unknown')

        # Determine overall status
        healthy_count = sum(1 for status in health_results.values() if status == 'healthy')
        total_services = len(health_results)

        if healthy_count == total_services:
            overall_status = 'healthy'
        elif healthy_count >= total_services * 0.8:  # 80% healthy
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'

        return {
            'overall_status': overall_status,
            'services': health_results,
            'healthy_services': healthy_count,
            'total_services': total_services,
            'health_percentage': (healthy_count / total_services) * 100,
            'checked_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"[{datetime.utcnow()}] ❌ Error in startup health checks: {str(e)}")
        return {
            'overall_status': 'error',
            'error': str(e)
        }

#=================================================================================================================================
# FASTAPI APPLICATION SETUP
#=================================================================================================================================

# Tạo FastAPI app với custom configuration
app = FastAPI(
    title="AI Challenge API",
    description="""
    # AI Challenge - Intelligent Video Processing & Chat System

    **Developed by Team xthanh1910**

    Một hệ thống AI tiên tiến cho video processing và conversational interactions.

    ## 🎯 Key Features

    ### 🎥 Video Processing
    - **Intelligent Upload**: Multi-format video upload với validation
    - **AI Analysis**: Automatic content analysis và feature extraction
    - **Smart Indexing**: Advanced video indexing cho search optimization
    - **Batch Processing**: Efficient processing của multiple videos

    ### 💬 Conversational AI
    - **Natural Chat**: Context-aware conversations với AI assistant
    - **Video Search**: Semantic search trong video content
    - **Smart Responses**: Intelligent responses với media references
    - **Session Management**: Persistent conversation contexts

    ### 👥 User Management
    - **Secure Authentication**: JWT-based auth với role management
    - **User Profiles**: Comprehensive user profile system
    - **Activity Tracking**: Detailed user activity analytics
    - **Admin Controls**: Advanced admin management features

    ### 🛡️ Security & Performance
    - **Rate Limiting**: Advanced rate limiting với burst support
    - **CORS Management**: Intelligent CORS handling
    - **Caching**: Multi-level caching cho performance
    - **Health Monitoring**: Real-time system health checks

    ## 🚀 Getting Started

    1. **Authentication**: Start với `/auth/login` hoặc `/auth/register`
    2. **Upload Videos**: Use `/videos/upload` để upload your content
    3. **Chat**: Interact với AI thông qua `/chat/message`
    4. **Search**: Find content với `/search` endpoints

    ## 📊 API Statistics

    - **Total Endpoints**: 50+ REST endpoints
    - **Real-time**: WebSocket support cho chat
    - **File Upload**: Multi-part upload với progress tracking
    - **Response Time**: < 200ms average response time

    ## 🔧 Technical Stack

    - **Framework**: FastAPI với async/await
    - **AI Agents**: Custom conversational và processing agents
    - **Database**: Advanced caching với Redis
    - **Security**: JWT, Rate limiting, CORS
    - **Performance**: Multi-threaded agent execution

    ---

    **Contact**: xthanh1910@ai-challenge.com
    **Version**: 1.0.0
    **Last Updated**: 2025-07-03 14:33:04 UTC
    """,
    version="1.0.0",
    contact={
        "name": "xthanh1910",
        "email": "xthanh1910@ai-challenge.com",
        "url": "https://github.com/xthanh1910/ai-challenge"
    },
    license_info={
        "name": "AI Challenge License",
        "url": "https://ai-challenge.com/license"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.ai-challenge.com",
            "description": "Production server"
        }
    ],
    docs_url=None,  # Disable default docs để custom
    redoc_url=None,  # Disable default redoc để custom
    openapi_url="/openapi.json",
    lifespan=lifespan  # Application lifecycle management
)

#=================================================================================================================================
# MIDDLEWARE CONFIGURATION
#=================================================================================================================================

# Bước 1: Trusted Host Middleware (security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "ai-challenge.com",
        "*.ai-challenge.com",
        "*.localhost"
    ]
)

# Bước 2: Session Middleware (cho admin sessions)
app.add_middleware(
    SessionMiddleware,
    secret_key="ai_challenge_session_secret_xthanh1910_change_in_production",
    max_age=86400,  # 24 hours
    https_only=False,  # Set True trong production
    same_site="lax"
)

# Bước 3: CORS Middleware (cho frontend integration)
cors_middleware = create_enhanced_cors_middleware(app, for_admin=False)

# Bước 4: Rate Limiting Middleware (security & performance)
rate_limit_manager = RateLimitManager()
app.add_middleware(RateLimitMiddleware, rate_manager=rate_limit_manager)

# Bước 5: Authentication Middleware (user management)
app.add_middleware(AuthenticationMiddleware)

# Bước 6: GZip Middleware (performance)
app.add_middleware(GZipMiddleware, minimum_size=1000)

#=================================================================================================================================
# EXCEPTION HANDLERS
#=================================================================================================================================

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler với structured responses
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": "HTTP_ERROR",
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(StarletteHTTPException)
async def custom_starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle Starlette HTTP exceptions
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": "STARLETTE_ERROR",
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """
    Handle internal server errors
    """
    print(f"[{datetime.utcnow()}] ❌ Internal Server Error: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error occurred",
            "error_code": "INTERNAL_SERVER_ERROR",
            "status_code": 500,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "The server encountered an unexpected error. Please try again later."
        }
    )

#=================================================================================================================================
# REQUEST/RESPONSE MIDDLEWARE
#=================================================================================================================================

@app.middleware("http")
async def request_processing_middleware(request: Request, call_next):
    """
    Middleware để process mỗi request và response
    """
    start_time = datetime.utcnow()
    request_id = f"req_{int(start_time.timestamp())}_{id(request)}"

    # Add request ID to state
    request.state.request_id = request_id

    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    print(f"[{start_time}] 📥 {request.method} {request.url.path} from {client_ip}")

    try:
        # Process request
        response = await call_next(request)

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Powered-By"] = "AI-Challenge-API"

        # Log response
        print(f"[{end_time}] 📤 {response.status_code} {request.method} {request.url.path} ({processing_time:.3f}s)")

        return response

    except Exception as e:
        # Log error
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        print(f"[{end_time}] ❌ Error processing {request.method} {request.url.path}: {str(e)} ({processing_time:.3f}s)")

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Request processing failed",
                "error_code": "REQUEST_PROCESSING_ERROR",
                "request_id": request_id,
                "timestamp": end_time.isoformat()
            }
        )

#=================================================================================================================================
# API ROUTERS
#=================================================================================================================================

# Include all API routers với prefixes
app.include_router(auth_router, prefix="/auth", tags=["🔐 Authentication"])
app.include_router(users_router, prefix="/users", tags=["👥 Users"])
app.include_router(videos_router, prefix="/videos", tags=["🎥 Videos"])
app.include_router(chat_router, prefix="/chat", tags=["💬 Chat"])
app.include_router(admin_router, prefix="/admin", tags=["🛡️ Admin"])
app.include_router(health_router, prefix="/health", tags=["🔍 Health"])

#=================================================================================================================================
# ROOT ENDPOINTS
#=================================================================================================================================

@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint - redirect to documentation
    """
    return RedirectResponse(url="/docs")

@app.get("/info", tags=["📋 Information"])
async def get_api_info():
    """
    Get API information và server status
    """
    try:
        startup_time = getattr(app.state, 'startup_time', datetime.utcnow())
        uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()

        # Get basic health status
        health_status = getattr(app.state, 'health_status', {})

        return {
            "success": True,
            "api_name": "AI Challenge API",
            "version": "1.0.0",
            "description": "Intelligent Video Processing & Chat System",
            "developer": "xthanh1910",
            "startup_time": startup_time.isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s",
            "current_time": datetime.utcnow().isoformat(),
            "timezone": "UTC",
            "server_status": "running",
            "health_status": health_status.get('overall_status', 'unknown'),
            "endpoints": {
                "documentation": "/docs",
                "alternative_docs": "/redoc",
                "openapi_spec": "/openapi.json",
                "health_check": "/health",
                "admin_panel": "/admin"
            },
            "features": [
                "Video Upload & Processing",
                "AI-Powered Chat",
                "Semantic Video Search",
                "User Management",
                "Admin Dashboard",
                "Real-time WebSocket",
                "Advanced Security"
            ],
            "statistics": {
                "total_endpoints": len([route for route in app.routes if hasattr(route, 'methods')]),
                "api_version": "1.0.0",
                "last_deployed": "2025-07-03T14:33:04Z"
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to get API info",
                "error_message": str(e)
            }
        )

#=================================================================================================================================
# CUSTOM DOCUMENTATION ENDPOINTS
#=================================================================================================================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI với enhanced configuration
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "operationsSorter": "alpha",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True
        }
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """
    Custom ReDoc documentation
    """
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Reference",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"
    )

@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    """
    Custom OpenAPI schema với enhanced metadata
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers
    )

    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://ai-challenge.com/logo.png",
        "altText": "AI Challenge Logo"
    }

    openapi_schema["info"]["x-api-id"] = "ai-challenge-api"
    openapi_schema["info"]["x-audience"] = "developers"

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /auth/login"
        }
    }

    # Add global security
    openapi_schema["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

#=================================================================================================================================
# STATIC FILES
#=================================================================================================================================

# Mount static files cho admin dashboard và assets
app.mount("/static", StaticFiles(directory="static"), name="static")

#=================================================================================================================================
# DEVELOPMENT UTILITIES
#=================================================================================================================================

if os.getenv("ENVIRONMENT", "development") == "development":

    @app.get("/dev/reset-cache", tags=["🔧 Development"])
    async def dev_reset_cache(current_user = Depends(get_current_user)):
        """
        Development utility: Reset all cache (admin only)
        """
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        try:
            result = await cache_service.emergency_clear_cache()
            return {
                "success": True,
                "message": "Cache reset completed",
                "result": result,
                "reset_by": current_user.username,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e)
                }
            )

    @app.get("/dev/system-info", tags=["🔧 Development"])
    async def dev_system_info():
        """
        Development utility: Get detailed system information
        """
        try:
            import psutil
            import sys

            # System info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "success": True,
                "system": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "percent": round((disk.used / disk.total) * 100, 2)
                    }
                },
                "application": {
                    "startup_time": getattr(app.state, 'startup_time', None),
                    "uptime_seconds": (datetime.utcnow() - getattr(app.state, 'startup_time', datetime.utcnow())).total_seconds(),
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Could not retrieve system info"
            }

#=================================================================================================================================
# APPLICATION ENTRY POINT
#=================================================================================================================================

def run_server():
    """
    Run the FastAPI server với uvicorn
    """
    # Set uvloop cho better performance (Linux/Mac only)
    if hasattr(asyncio, 'set_event_loop_policy'):
        try:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print(f"[{datetime.utcnow()}] ⚡ Using uvloop for enhanced performance")
        except Exception:
            print(f"[{datetime.utcnow()}] ℹ️  Using default event loop (uvloop not available)")

    # Server configuration
    config = {
        "app": "api.main:app",
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8000)),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
        "access_log": True,
        "workers": 1 if os.getenv("ENVIRONMENT", "development") == "development" else 4,
        "loop": "uvloop" if hasattr(asyncio, 'set_event_loop_policy') else "asyncio"
    }

    print(f"[{datetime.utcnow()}] 🌟 AI Challenge API Server Configuration:")
    print(f"[{datetime.utcnow()}] 📍 Host: {config['host']}")
    print(f"[{datetime.utcnow()}] 🔌 Port: {config['port']}")
    print(f"[{datetime.utcnow()}] 🔄 Reload: {config['reload']}")
    print(f"[{datetime.utcnow()}] 👷 Workers: {config['workers']}")
    print(f"[{datetime.utcnow()}] 🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")

    # Start server
    uvicorn.run(**config)

#=================================================================================================================================
# EXPORTS
#=================================================================================================================================

# Export app instance for ASGI servers
__all__ = ["app", "run_server"]

if __name__ == "__main__":
    run_server()

print(f"[{datetime.utcnow()}] 📚 AI Challenge API Main Module loaded by user: xthanh1910")