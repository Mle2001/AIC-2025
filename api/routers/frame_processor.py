"""
Frame Sequence Processing API Routes
===================================

API endpoints for frame sequence processing and competition queries.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time

from api.services.minimal_frame_processor_service import minimal_frame_processor_service
from pydantic import BaseModel

router = APIRouter(prefix="/api/frame-processor", tags=["frame-processor"])

# Pydantic models for request/response
class FrameSequenceUploadRequest(BaseModel):
    folder_name: str
    fps: Optional[float] = 30.0

class FrameSequenceUploadResponse(BaseModel):
    status: str
    folder_name: str
    frame_count: int
    folder_path: str
    message: str

class ProcessingStatusResponse(BaseModel):
    status: str
    total_sequences: Optional[int] = None
    successful_sequences: Optional[int] = None
    failed_sequences: Optional[int] = None
    last_updated: Optional[str] = None
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    fast_mode: bool = True

class QueryResponse(BaseModel):
    status: str
    query: str
    video_folder: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    confidence: float
    query_time: float
    raw_response: Optional[str] = None
    error: Optional[str] = None

class QueryStatsResponse(BaseModel):
    total_queries: int
    average_query_time: float
    success_rate: float
    mode: str

@router.post("/upload-sequence", response_model=FrameSequenceUploadResponse)
async def upload_frame_sequence(
    folder_name: str,
    fps: Optional[float] = 30.0,
    frame_files: List[UploadFile] = File(...)
):
    """
    Upload frame sequence files for processing
    
    Args:
        folder_name: Name of the video folder
        fps: Frames per second for the sequence
        frame_files: List of frame image files
        
    Returns:
        Upload result
    """
    try:
        # Validate frame files
        if not frame_files:
            raise HTTPException(status_code=400, detail="No frame files provided")
        
        # Check file types
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for frame_file in frame_files:
            if not any(frame_file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {frame_file.filename}. Allowed: {allowed_extensions}"
                )
        
        # Upload frame sequence
        result = await minimal_frame_processor_service.upload_frame_sequence(
            folder_name=folder_name,
            frame_files=frame_files
        )
        
        if result['status'] == 'failed':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return FrameSequenceUploadResponse(
            status=result['status'],
            folder_name=result['folder_name'],
            frame_count=result['frame_count'],
            folder_path=result['folder_path'],
            message=f"Successfully uploaded {result['frame_count']} frames"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading frame sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-sequence/{folder_name}")
async def process_frame_sequence(
    folder_name: str,
    fps: Optional[float] = 30.0
):
    """
    Process a specific frame sequence folder
    
    Args:
        folder_name: Name of the folder containing frames
        fps: Frames per second for processing
        
    Returns:
        Processing result
    """
    try:
        result = await minimal_frame_processor_service.process_frame_sequence(
            folder_name=folder_name,
            fps=fps
        )
        
        if result['status'] == 'failed':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing frame sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/build-index")
async def build_videorag_index():
    """
    Build VideoRAG index from all processed frame sequences
    
    Returns:
        Index building result
    """
    try:
        result = await minimal_frame_processor_service.build_videorag_index()
        
        if result['status'] == 'failed':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error building VideoRAG index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_frame_sequences(request: QueryRequest):
    """
    Query processed frame sequences
    
    Args:
        request: Query request with query text and options
        
    Returns:
        Query result
    """
    try:
        result = await minimal_frame_processor_service.query_frame_sequences(
            query=request.query,
            fast_mode=request.fast_mode
        )
        
        return QueryResponse(
            status=result['status'],
            query=result['query'],
            video_folder=result.get('video_folder'),
            start_timestamp=result.get('start_timestamp'),
            end_timestamp=result.get('end_timestamp'),
            confidence=result.get('confidence', 0.0),
            query_time=result.get('query_time', 0.0),
            raw_response=result.get('raw_response'),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error querying frame sequences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=ProcessingStatusResponse)
async def get_processing_status():
    """
    Get current processing status
    
    Returns:
        Processing status information
    """
    try:
        result = await minimal_frame_processor_service.get_processing_status()
        
        return ProcessingStatusResponse(
            status=result['status'],
            total_sequences=result.get('total_sequences'),
            successful_sequences=result.get('successful_sequences'),
            failed_sequences=result.get('failed_sequences'),
            last_updated=result.get('last_updated'),
            message=result.get('message')
        )
        
    except Exception as e:
        logging.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/folders")
async def get_frame_folders():
    """
    Get list of available frame sequence folders
    
    Returns:
        List of folder names
    """
    try:
        folders = await minimal_frame_processor_service.discover_frame_folders()
        
        return JSONResponse(content={
            'folders': folders,
            'count': len(folders)
        })
        
    except Exception as e:
        logging.error(f"Error getting frame folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query-stats", response_model=QueryStatsResponse)
async def get_query_stats():
    """
    Get query performance statistics
    
    Returns:
        Query statistics
    """
    try:
        stats = await minimal_frame_processor_service.get_query_stats()
        
        if 'message' in stats:
            return JSONResponse(content=stats)
        
        return QueryStatsResponse(
            total_queries=stats['total_queries'],
            average_query_time=stats['average_query_time'],
            success_rate=stats['success_rate'],
            mode=stats['mode']
        )
        
    except Exception as e:
        logging.error(f"Error getting query stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_temp_files():
    """
    Clean up temporary video files
    
    Returns:
        Cleanup result
    """
    try:
        await minimal_frame_processor_service.cleanup_temp_files()
        
        return JSONResponse(content={
            'status': 'success',
            'message': 'Temporary files cleaned up successfully'
        })
        
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-process")
async def batch_process_sequences(
    fps: Optional[float] = 30.0,
    build_index: bool = True
):
    """
    Process all discovered frame sequences in batch
    
    Args:
        fps: Frames per second for all sequences
        build_index: Whether to build VideoRAG index after processing
        
    Returns:
        Batch processing result
    """
    try:
        # Discover frame folders
        folders = await minimal_frame_processor_service.discover_frame_folders()
        
        if not folders:
            return JSONResponse(content={
                'status': 'no_folders',
                'message': 'No frame sequence folders found'
            })
        
        # Process each folder
        results = []
        for folder in folders:
            result = await minimal_frame_processor_service.process_frame_sequence(
                folder_name=folder,
                fps=fps
            )
            results.append(result)
        
        # Build index if requested
        index_result = None
        if build_index:
            index_result = await minimal_frame_processor_service.build_videorag_index()
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        return JSONResponse(content={
            'status': 'completed',
            'total_folders': len(folders),
            'successful': successful,
            'failed': failed,
            'results': results,
            'index_result': index_result
        })
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    try:
        # Check basic functionality
        folders = await minimal_frame_processor_service.discover_frame_folders()
        status = await minimal_frame_processor_service.get_processing_status()
        
        return JSONResponse(content={
            'status': 'healthy',
            'service': 'frame_processor',
            'folders_available': len(folders),
            'processing_status': status['status'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Competition-specific endpoints
@router.post("/competition/quick-query")
async def competition_quick_query(
    query: str = Query(..., description="Query text"),
    timeout: float = Query(10.0, description="Query timeout in seconds")
):
    """
    Quick query for competition use
    
    Args:
        query: Search query text
        timeout: Query timeout
        
    Returns:
        Formatted competition result
    """
    try:
        result = await minimal_frame_processor_service.query_frame_sequences(
            query=query,
            fast_mode=True
        )
        
        if result['status'] != 'success':
            return JSONResponse(content={
                'status': 'error',
                'message': result.get('error', 'Query failed')
            })
        
        # Format for competition
        video_folder = result.get('video_folder', 'unknown')
        start_time = result.get('start_timestamp', 0.0)
        end_time = result.get('end_timestamp', 10.0)
        
        def format_time(seconds):
            if seconds is None:
                return "00:00"
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        formatted_result = f"Video: {video_folder}, Time: {format_time(start_time)} - {format_time(end_time)}"
        
        return JSONResponse(content={
            'status': 'success',
            'query': query,
            'result': formatted_result,
            'query_time': result.get('query_time', 0.0),
            'confidence': result.get('confidence', 0.0),
            'details': {
                'video_folder': video_folder,
                'start_timestamp': start_time,
                'end_timestamp': end_time
            }
        })
        
    except Exception as e:
        logging.error(f"Error in competition query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/competition/batch-query")
async def competition_batch_query(
    queries: List[str],
    output_format: str = Query("json", description="Output format: json or text")
):
    """
    Batch query for competition
    
    Args:
        queries: List of query strings
        output_format: Output format (json or text)
        
    Returns:
        Batch query results
    """
    try:
        results = []
        
        for query in queries:
            result = await minimal_frame_processor_service.query_frame_sequences(
                query=query,
                fast_mode=True
            )
            
            if result['status'] == 'success':
                video_folder = result.get('video_folder', 'unknown')
                start_time = result.get('start_timestamp', 0.0)
                end_time = result.get('end_timestamp', 10.0)
                
                def format_time(seconds):
                    if seconds is None:
                        return "00:00"
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{minutes:02d}:{secs:02d}"
                
                formatted_result = f"Video: {video_folder}, Time: {format_time(start_time)} - {format_time(end_time)}"
                
                results.append({
                    'query': query,
                    'result': formatted_result,
                    'query_time': result.get('query_time', 0.0),
                    'confidence': result.get('confidence', 0.0)
                })
            else:
                results.append({
                    'query': query,
                    'result': f"ERROR: {result.get('error', 'Query failed')}",
                    'query_time': result.get('query_time', 0.0),
                    'confidence': 0.0
                })
        
        if output_format == "text":
            text_results = [r['result'] for r in results]
            return JSONResponse(content={
                'status': 'success',
                'format': 'text',
                'results': text_results
            })
        else:
            return JSONResponse(content={
                'status': 'success',
                'format': 'json',
                'results': results
            })
        
    except Exception as e:
        logging.error(f"Error in batch query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
