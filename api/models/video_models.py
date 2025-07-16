from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime

class VideoUploadRequest(BaseModel):
    """Request model for video upload"""
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for processing")
    auto_process: Optional[bool] = Field(False, description="Auto process video after upload")

class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Upload status")
    message: Optional[str] = Field(None, description="Additional message")

class VideoProcessRequest(BaseModel):
    """Request model for video processing"""
    file_id: str = Field(..., description="File ID to process")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for processing (optional, uses backend default if not provided)")

class VideoProcessResponse(BaseModel):
    """Response model for video processing"""
    file_id: str = Field(..., description="File identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Processing message")
    error: Optional[str] = Field(None, description="Error message if any")

class VideoQueryRequest(BaseModel):
    """Request model for video query"""
    file_id: str = Field(..., description="File ID to query")
    query: str = Field(..., description="Query text")
    openai_api_key: str = Field(..., description="OpenAI API key")

class VideoQueryResponse(BaseModel):
    """Response model for video query"""
    file_id: str = Field(..., description="File identifier")
    query: str = Field(..., description="Query text")
    response: str = Field(..., description="AI response")
    status: str = Field(..., description="Query status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class VideoInfo(BaseModel):
    """Video information model"""
    file_id: str = Field(..., description="Unique file identifier")
    original_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    user_id: str = Field(..., description="Owner user ID")
    status: str = Field(..., description="Processing status")
    processed: bool = Field(..., description="Whether video is processed")
    error: Optional[str] = Field(None, description="Error message if any")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")

class VideoListResponse(BaseModel):
    """Response model for video list"""
    videos: List[VideoInfo] = Field(..., description="List of user videos")
    total: int = Field(..., description="Total number of videos")

class VideoDeleteResponse(BaseModel):
    """Response model for video deletion"""
    file_id: str = Field(..., description="Deleted file identifier")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Deletion message")

class VideoResponse(BaseModel):
    """General video response model"""
    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")