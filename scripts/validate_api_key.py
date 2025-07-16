#!/usr/bin/env python3
"""
Script để test OpenAI API key
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_openai_api_key():
    """Test OpenAI API key"""
    try:
        import openai
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "your-openai-api-key-here":
            print("❌ OpenAI API key chưa được set trong file .env")
            return False
        
        # Set API key
        openai.api_key = api_key
        
        # Test API call
        print("🔍 Testing OpenAI API key...")
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API key hoạt động tốt!")
        print(f"📝 Response: {response.choices[0].message.content}")
        return True
        
    except ImportError:
        print("❌ OpenAI library chưa được cài đặt")
        print("💡 Chạy: pip install openai")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi test OpenAI API key: {e}")
        return False

def check_env_file():
    """Kiểm tra file .env"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("❌ File .env không tồn tại")
        return False
    
    print("✅ File .env tồn tại")
    
    # Kiểm tra các biến môi trường quan trọng
    required_vars = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL", 
        "OPENAI_EMBEDDING_MODEL"
    ]
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    for var in required_vars:
        if var in content:
            print(f"✅ {var} có trong file .env")
        else:
            print(f"❌ {var} thiếu trong file .env")
    
    return True

if __name__ == "__main__":
    print("🔍 Kiểm tra cấu hình OpenAI API...")
    print("=" * 50)
    
    # Kiểm tra file .env
    check_env_file()
    print()
    
    # Test API key
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_openai_api_key())
    else:
        print("💡 Để test API key, chạy: python scripts/validate_api_key.py --test")
