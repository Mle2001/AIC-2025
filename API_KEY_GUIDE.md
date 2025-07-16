# API Key Management

## 🔑 Cách thêm OpenAI API Key vào hệ thống

### 1. Lấy OpenAI API Key

1. Truy cập [OpenAI Platform](https://platform.openai.com)
2. Đăng nhập vào tài khoản của bạn
3. Vào **API Keys** section
4. Tạo new API key
5. Copy API key (format: `sk-...`)

### 2. Thêm API Key vào hệ thống

#### Cách 1: Sử dụng script helper (Khuyến nghị)

```bash
python scripts/set_api_key.py sk-your-actual-openai-api-key-here
```

#### Cách 2: Chỉnh sửa file .env trực tiếp

Mở file `.env` và thay đổi:
```env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

#### Cách 3: Set biến môi trường

**Windows:**
```cmd
set OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### 3. Kiểm tra API Key

```bash
# Kiểm tra API key trong file .env
python scripts/validate_api_key.py

# Test API key với OpenAI
python scripts/validate_api_key.py --test
```

## 🏗️ Cấu trúc cấu hình

### File .env
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### File api/config.py
```python
class Settings(BaseModel):
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
```

## 🔍 Troubleshooting

### API Key không hoạt động
1. Kiểm tra API key format (phải bắt đầu bằng `sk-`)
2. Kiểm tra API key còn active trên OpenAI Platform
3. Kiểm tra credit/quota trên OpenAI account

### Environment variables không load
1. Restart server sau khi thay đổi .env
2. Kiểm tra file .env có trong đúng thư mục gốc
3. Kiểm tra python-dotenv đã được cài đặt

### API calls bị lỗi
1. Kiểm tra internet connection
2. Kiểm tra OpenAI service status
3. Kiểm tra rate limits

## 📋 Scripts hỗ trợ

- `scripts/set_api_key.py` - Set OpenAI API key
- `scripts/validate_api_key.py` - Validate và test API key

## 🔒 Bảo mật

- **KHÔNG** commit API key vào git
- **KHÔNG** share API key
- **KHÔNG** hard-code API key trong source code
- **SỬ DỤNG** environment variables hoặc .env file
- **THÊM** .env vào .gitignore
