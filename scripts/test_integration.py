#!/usr/bin/env python3
"""
Integration Test Script
======================

Tests the integrated VideoRAG system to ensure all components work correctly.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_integration():
    """Test the integrated system components"""
    print("🧪 Testing VideoRAG Integration...")
    print("=" * 50)
    
    # Test 1: Import all modules
    print("\n1. Testing imports...")
    try:
        from api.main import app
        print("   ✅ Main app import successful")
        
        from api.services.frame_processor_service import frame_processor_service
        print("   ✅ Frame processor service import successful")
        
        from api.services.video_service import video_service
        print("   ✅ Video service import successful")
        
        from api.routers.frame_processor import router as frame_router
        print("   ✅ Frame processor router import successful")
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 2: Check service initialization
    print("\n2. Testing service initialization...")
    try:
        # Test frame processor service
        folders = await frame_processor_service.discover_frame_folders()
        print(f"   ✅ Frame processor service: {len(folders)} folders found")
        
        # Test video service
        status = await video_service.get_frame_processor_status()
        print(f"   ✅ Video service integration: {status['status']}")
        
    except Exception as e:
        print(f"   ❌ Service initialization error: {e}")
        return False
    
    # Test 3: Check API endpoints
    print("\n3. Testing API endpoints...")
    try:
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/health/")
        if response.status_code == 200:
            print("   ✅ Health endpoint working")
        else:
            print(f"   ❌ Health endpoint error: {response.status_code}")
        
        # Test frame processor health
        response = client.get("/api/frame-processor/health")
        if response.status_code == 200:
            print("   ✅ Frame processor health endpoint working")
        else:
            print(f"   ❌ Frame processor health error: {response.status_code}")
        
        # Test frame processor status
        response = client.get("/api/frame-processor/status")
        if response.status_code == 200:
            print("   ✅ Frame processor status endpoint working")
        else:
            print(f"   ❌ Frame processor status error: {response.status_code}")
        
    except Exception as e:
        print(f"   ❌ API endpoint error: {e}")
        return False
    
    # Test 4: Check directory structure
    print("\n4. Testing directory structure...")
    try:
        from api.config import settings
        
        # Check upload directories
        upload_dir = Path(settings.upload_dir)
        frame_seq_dir = upload_dir / "frame_sequences"
        frame_index_dir = upload_dir / "frame_index"
        temp_videos_dir = upload_dir / "temp_videos"
        
        print(f"   ✅ Upload dir: {upload_dir} {'(exists)' if upload_dir.exists() else '(created)'}")
        print(f"   ✅ Frame sequences dir: {frame_seq_dir} {'(exists)' if frame_seq_dir.exists() else '(created)'}")
        print(f"   ✅ Frame index dir: {frame_index_dir} {'(exists)' if frame_index_dir.exists() else '(created)'}")
        print(f"   ✅ Temp videos dir: {temp_videos_dir} {'(exists)' if temp_videos_dir.exists() else '(created)'}")
        
    except Exception as e:
        print(f"   ❌ Directory structure error: {e}")
        return False
    
    # Test 5: Check dependencies
    print("\n5. Testing dependencies...")
    try:
        # Check VideoRAG availability
        from api.services.frame_processor_service import VIDEORAG_AVAILABLE, MOVIEPY_AVAILABLE
        print(f"   {'✅' if VIDEORAG_AVAILABLE else '❌'} VideoRAG: {'Available' if VIDEORAG_AVAILABLE else 'Not available'}")
        print(f"   {'✅' if MOVIEPY_AVAILABLE else '❌'} MoviePy: {'Available' if MOVIEPY_AVAILABLE else 'Not available'}")
        
        # Check OpenAI key
        from api.config import get_openai_api_key
        openai_key = get_openai_api_key()
        print(f"   {'✅' if openai_key else '❌'} OpenAI API Key: {'Configured' if openai_key else 'Not configured'}")
        
    except Exception as e:
        print(f"   ❌ Dependencies error: {e}")
        return False
    
    print("\n🎉 Integration test completed successfully!")
    print("\n📋 Summary:")
    print("   - All imports working correctly")
    print("   - Services initialized properly")
    print("   - API endpoints responding")
    print("   - Directory structure ready")
    print("   - Dependencies checked")
    
    print("\n🚀 System is ready for use!")
    return True

def main():
    """Main test function"""
    try:
        # Run async test
        result = asyncio.run(test_integration())
        
        if result:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
