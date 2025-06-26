from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
import os
from api.routers import chat, upload, admin, health

# Middleware setup function
def setup_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Có thể add thêm các middleware khác như auth, rate_limit ở đây

# Routers setup function
def setup_routes(app: FastAPI):
    app.include_router(chat.router, prefix="/api/chat")
    app.include_router(upload.router, prefix="/api/upload")
    app.include_router(admin.router, prefix="/api/admin")
    app.include_router(health.router, prefix="/api/health")

# Factory function tạo FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(title="Agno AI Backend")
    setup_middleware(app)
    setup_routes(app)
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app

app = create_app()

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    index_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
