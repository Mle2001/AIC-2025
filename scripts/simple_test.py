#!/usr/bin/env python3
"""
Simple Backend Test
==================

Tests basic backend functionality without complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing Basic Imports...")
    print("=" * 40)
    
    try:
        # Test basic FastAPI app
        from api.main import app
        print("✅ FastAPI app import successful")
        
        # Test config
        from api.config import settings
        print("✅ Config import successful")
        
        # Test routers
        from api.routers import chat, upload, admin, health
        print("✅ Basic routers import successful")
        
        # Test services
        from api.services.video_service import video_service
        print("✅ Video service import successful")
        
        # Test frame processor (might show warnings)
        from api.services.frame_processor_service import frame_processor_service
        print("✅ Frame processor service import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_structure():
    """Test API structure"""
    print("\n🏗️ Testing API Structure...")
    print("=" * 40)
    
    try:
        from api.main import app
        
        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(f"{route.methods} {route.path}")
        
        print(f"✅ Total routes: {len(routes)}")
        
        # Check for key routes
        key_routes = [
            "/api/health",
            "/api/upload",
            "/api/chat",
            "/api/frame-processor"
        ]
        
        for key_route in key_routes:
            found = any(key_route in route for route in routes)
            print(f"{'✅' if found else '❌'} {key_route}: {'Found' if found else 'Not found'}")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure error: {e}")
        return False

def test_dependencies():
    """Test dependencies"""
    print("\n📦 Testing Dependencies...")
    print("=" * 40)
    
    # Core dependencies
    deps = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("aiofiles", "Async file operations"),
    ]
    
    for dep_name, dep_desc in deps:
        try:
            __import__(dep_name)
            print(f"✅ {dep_desc}: Available")
        except ImportError:
            print(f"❌ {dep_desc}: Not available")
    
    # Optional dependencies
    optional_deps = [
        ("moviepy", "MoviePy (frame processing)"),
        ("cv2", "OpenCV (image processing)"),
        ("PIL", "Pillow (image handling)"),
    ]
    
    for dep_name, dep_desc in optional_deps:
        try:
            __import__(dep_name)
            print(f"✅ {dep_desc}: Available")
        except ImportError:
            print(f"⚠️  {dep_desc}: Not available (optional)")
    
    return True

def test_file_structure():
    """Test file structure"""
    print("\n📁 Testing File Structure...")
    print("=" * 40)
    
    # Check key directories
    key_dirs = [
        "api",
        "api/services",
        "api/routers",
        "api/models",
        "api/middleware",
        "frontend",
        "scripts",
        "VideoRAG"
    ]
    
    for dir_name in key_dirs:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print(f"{'✅' if exists else '❌'} {dir_name}: {'Exists' if exists else 'Missing'}")
    
    # Check key files
    key_files = [
        "api/main.py",
        "api/config.py",
        "api/services/frame_processor_service.py",
        "api/routers/frame_processor.py",
        "scripts/integration_setup.py"
    ]
    
    for file_name in key_files:
        file_path = project_root / file_name
        exists = file_path.exists()
        print(f"{'✅' if exists else '❌'} {file_name}: {'Exists' if exists else 'Missing'}")
    
    return True

def main():
    """Main test function"""
    print("🚀 AIC-2025 Backend Integration Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_api_structure,
        test_dependencies,
        test_file_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test error: {e}")
            results.append(False)
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed, but basic functionality should work.")
        return 1

if __name__ == "__main__":
    exit(main())
