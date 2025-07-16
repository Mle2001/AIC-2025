#!/usr/bin/env python3
"""
Integration Setup Script
========================

Moves original VideoRAG files to avoid conflicts and sets up integrated system.
"""

import os
import shutil
from pathlib import Path

def main():
    """Main setup function"""
    project_root = Path(__file__).parent
    
    # Create backup directory
    backup_dir = project_root / "scripts" / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Move original files to backup
    files_to_backup = [
        "phase2_competition.py",
        "videorag_frame_adapter.py"
    ]
    
    for file_name in files_to_backup:
        original_file = project_root / file_name
        backup_file = backup_dir / file_name
        
        if original_file.exists():
            if backup_file.exists():
                backup_file.unlink()
            shutil.move(str(original_file), str(backup_file))
            print(f"✅ Moved {file_name} to backup")
        else:
            print(f"⚠️  {file_name} not found")
    
    # Create integration info file
    integration_info = backup_dir / "integration_info.md"
    with open(integration_info, 'w', encoding='utf-8') as f:
        f.write("""# VideoRAG Integration Info

## Original Files Moved

- `phase2_competition.py` -> Integrated into `api/services/frame_processor_service.py`
- `videorag_frame_adapter.py` -> Integrated into `api/services/frame_processor_service.py`

## New API Endpoints

### Frame Processor Service (`/api/frame-processor/`)
- `POST /upload-sequence` - Upload frame sequence
- `POST /process-sequence/{folder_name}` - Process frame sequence
- `POST /build-index` - Build VideoRAG index
- `POST /query` - Query frame sequences
- `GET /status` - Get processing status
- `GET /folders` - List frame folders
- `GET /query-stats` - Get query statistics
- `DELETE /cleanup` - Clean up temporary files

### Upload Service (`/api/upload/`)
- `POST /frame-sequence` - Upload frame sequence (with auto-process)
- `POST /frame-sequence/{folder_name}/process` - Process frame sequence
- `POST /frame-sequence/query` - Query frame sequences
- `POST /frame-sequence/build-index` - Build index
- `GET /frame-sequence/status` - Get status
- `DELETE /frame-sequence/cleanup` - Cleanup

### Competition Endpoints (`/api/frame-processor/competition/`)
- `POST /quick-query` - Quick competition query
- `POST /batch-query` - Batch competition queries

## Integration Benefits

1. **Unified API** - All functionality accessible through REST endpoints
2. **Async Processing** - Non-blocking frame sequence processing
3. **Competition Ready** - Optimized queries for competition use
4. **Integrated Storage** - Unified file and metadata management
5. **Error Handling** - Comprehensive error handling and logging
6. **Performance Monitoring** - Built-in query statistics and performance tracking

## Usage Examples

### Upload and Process Frame Sequence
```bash
curl -X POST "http://localhost:8000/api/upload/frame-sequence" \
  -F "folder_name=video1" \
  -F "fps=30.0" \
  -F "auto_process=true" \
  -F "frame_files=@frame1.jpg" \
  -F "frame_files=@frame2.jpg"
```

### Query Frame Sequences
```bash
curl -X POST "http://localhost:8000/api/frame-processor/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find cooking scenes", "fast_mode": true}'
```

### Competition Query
```bash
curl -X POST "http://localhost:8000/api/frame-processor/competition/quick-query?query=Find%20cooking%20scenes"
```

## Migration Path

1. **Backup Complete** - Original files moved to backup
2. **Integration Complete** - New services integrated
3. **API Endpoints Ready** - All endpoints available
4. **Competition Ready** - Optimized for competition use

## Next Steps

1. Test the integrated system
2. Update frontend to use new endpoints
3. Run performance benchmarks
4. Deploy to production environment
""")
    
    print("\n🎉 Integration setup complete!")
    print(f"📁 Backup files saved to: {backup_dir}")
    print(f"📖 Integration info: {integration_info}")
    print("\n🚀 Ready to use integrated VideoRAG system!")

if __name__ == "__main__":
    main()
