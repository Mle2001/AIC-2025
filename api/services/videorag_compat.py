"""
Safe VideoRAG imports with Windows compatibility
"""

import sys
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

def safe_import_videorag():
    """Safely import VideoRAG with fallback handling"""
    try:
        # Add VideoRAG to path
        videorag_path = os.path.join(os.path.dirname(__file__), '..', '..', 'VideoRAG')
        if videorag_path not in sys.path:
            sys.path.append(videorag_path)
        
        # Try to import VideoRAG components
        from videorag.videorag import VideoRAG, QueryParam
        from videorag._llm import LLMConfig, openai_embedding, gpt_4o_mini_complete
        
        return {
            'available': True,
            'VideoRAG': VideoRAG,
            'QueryParam': QueryParam,
            'LLMConfig': LLMConfig,
            'openai_embedding': openai_embedding,
            'gpt_4o_mini_complete': gpt_4o_mini_complete,
            'error': None
        }
    
    except ImportError as e:
        logging.warning(f"VideoRAG import error: {e}")
        return {'available': False, 'error': str(e)}
    except OSError as e:
        logging.warning(f"VideoRAG OS error (likely torchaudio): {e}")
        return {'available': False, 'error': f"OS compatibility issue: {str(e)}"}
    except Exception as e:
        logging.warning(f"VideoRAG unexpected error: {e}")
        return {'available': False, 'error': str(e)}

# Create fallback classes
class MockVideoRAG:
    def __init__(self, *args, **kwargs):
        raise ImportError("VideoRAG not available due to dependency issues")

class MockQueryParam:
    def __init__(self, *args, **kwargs):
        raise ImportError("VideoRAG not available due to dependency issues")

class MockLLMConfig:
    def __init__(self, *args, **kwargs):
        raise ImportError("VideoRAG not available due to dependency issues")

# Try to import VideoRAG
videorag_result = safe_import_videorag()
VIDEORAG_AVAILABLE = videorag_result['available']

if VIDEORAG_AVAILABLE:
    VideoRAG = videorag_result['VideoRAG']
    QueryParam = videorag_result['QueryParam']
    LLMConfig = videorag_result['LLMConfig']
    openai_embedding = videorag_result['openai_embedding']
    gpt_4o_mini_complete = videorag_result['gpt_4o_mini_complete']
else:
    VideoRAG = MockVideoRAG
    QueryParam = MockQueryParam
    LLMConfig = MockLLMConfig
    openai_embedding = None
    gpt_4o_mini_complete = None
    
print(f"VideoRAG Status: {'Available' if VIDEORAG_AVAILABLE else 'Not Available'}")
if not VIDEORAG_AVAILABLE:
    print(f"Error: {videorag_result['error']}")
