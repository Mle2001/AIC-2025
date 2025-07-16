#!/usr/bin/env python3
"""
Script helper để set OpenAI API key
"""
import os
import sys

def set_openai_api_key(api_key: str):
    """Set OpenAI API key trong file .env"""
    env_file = ".env"
    
    # Đọc file .env hiện tại
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Tìm và thay thế dòng OPENAI_API_KEY
    found = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[i] = f"OPENAI_API_KEY={api_key}\n"
            found = True
            break
    
    # Nếu không tìm thấy, thêm mới
    if not found:
        lines.append(f"OPENAI_API_KEY={api_key}\n")
    
    # Ghi lại file .env
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ OpenAI API key đã được set thành công!")
    print(f"📝 File .env đã được cập nhật")

def get_openai_api_key():
    """Lấy OpenAI API key từ file .env"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("❌ File .env không tồn tại")
        return None
    
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "your-openai-api-key-here":
                    print(f"✅ OpenAI API key hiện tại: {key[:10]}...{key[-4:]}")
                    return key
                else:
                    print("⚠️  OpenAI API key chưa được set")
                    return None
    
    print("❌ Không tìm thấy OPENAI_API_KEY trong file .env")
    return None

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔍 Kiểm tra OpenAI API key hiện tại:")
        get_openai_api_key()
        print("\n📋 Cách sử dụng:")
        print("  python scripts/set_api_key.py <your-api-key>")
        print("  python scripts/set_api_key.py sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    elif len(sys.argv) == 2:
        api_key = sys.argv[1]
        if api_key.startswith("sk-"):
            set_openai_api_key(api_key)
        else:
            print("❌ API key không hợp lệ. OpenAI API key phải bắt đầu bằng 'sk-'")
    else:
        print("❌ Cách sử dụng: python scripts/set_api_key.py <your-api-key>")
