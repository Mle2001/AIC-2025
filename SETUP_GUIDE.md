# Hướng dẫn cài đặt và chạy hệ thống AIC-2025 với VideoRAG

## Yêu cầu hệ thống

- Python 3.8+
- Node.js 16+
- npm hoặc yarn
- Git

## Cài đặt Backend

1. **Cài đặt dependencies Python:**
```bash
pip install fastapi uvicorn python-multipart aiofiles python-dotenv openai pydantic
```

2. **Cài đặt VideoRAG dependencies:**
```bash
pip install torch torchvision torchaudio opencv-python transformers
```

3. **Cấu hình OpenAI API Key:**

### Cách 1: Sử dụng script helper (Khuyến nghị)
```bash
python scripts/set_api_key.py sk-your-actual-openai-api-key-here
```

### Cách 2: Chỉnh sửa file .env trực tiếp
- Mở file `.env` trong thư mục gốc
- Thay `sk-your-actual-openai-api-key-here` bằng API key OpenAI thực tế của bạn:
```
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### Cách 3: Set biến môi trường (Windows)
```cmd
set OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### Cách 4: Set biến môi trường (Linux/Mac)
```bash
export OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### Kiểm tra API key
```bash
python scripts/validate_api_key.py --test
```

4. **Tạo thư mục uploads:**
```bash
mkdir uploads
```

## Cài đặt Frontend

1. **Di chuyển vào thư mục frontend:**
```bash
cd frontend
```

2. **Cài đặt dependencies:**
```bash
npm install
```

## Chạy ứng dụng

### 1. Chạy Backend Server

Từ thư mục gốc của project:

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại: http://localhost:8000

### 2. Chạy Frontend

Mở terminal mới, từ thư mục frontend:

```bash
npm run dev
```

Frontend sẽ chạy tại: http://localhost:5173

## Cách sử dụng

### 1. Truy cập Admin Dashboard

- Mở trình duyệt và truy cập: http://localhost:5173
- Bạn sẽ thấy Admin Dashboard với:
  - Khu vực chat bên trái
  - Danh sách video bên phải
  - Tabs quản lý (Video Upload, Video List, Video Chat)

### 2. Upload Video

- Click tab "Video Upload"
- Chọn file video (hỗ trợ: mp4, avi, mov, mkv, webm)
- Nhấn "Upload Video"
- Hệ thống sẽ xử lý video và tạo embeddings

### 3. Chat với Video

- Click tab "Video Chat"
- Chọn video từ danh sách
- Nhập câu hỏi về nội dung video
- Hệ thống sẽ phân tích và trả lời dựa trên nội dung video

### 4. Chat thông thường

- Sử dụng khu vực chat bên trái
- Nhập câu hỏi bất kỳ
- Hệ thống sẽ trả lời sử dụng OpenAI API

## Cấu trúc API

### Chat Endpoints
- `POST /api/chat/message` - Gửi tin nhắn chat
- `GET /api/chat/videos` - Lấy danh sách video

### Video Endpoints
- `POST /api/upload/video` - Upload video
- `POST /api/upload/query` - Truy vấn video
- `GET /api/upload/videos` - Lấy danh sách video

### Health Check
- `GET /api/health` - Kiểm tra trạng thái server

## Troubleshooting

### Backend không khởi động được:
1. Kiểm tra Python version: `python --version`
2. Kiểm tra dependencies: `pip list`
3. Kiểm tra file .env có đúng format không
4. Kiểm tra port 8000 có bị chiếm không

### Frontend không khởi động được:
1. Kiểm tra Node.js version: `node --version`
2. Xóa node_modules và cài lại: `rm -rf node_modules && npm install`
3. Kiểm tra port 5173 có bị chiếm không

### Video processing lỗi:
1. Kiểm tra VideoRAG model có tồn tại không
2. Kiểm tra OpenAI API key
3. Kiểm tra file video có format hỗ trợ không

## Cấu hình nâng cao

### Thay đổi OpenAI model:
Trong file `.env`, thay đổi:
```
OPENAI_MODEL=gpt-4  # hoặc model khác
```

### Thay đổi kích thước file upload:
Trong file `.env`, thay đổi:
```
MAX_FILE_SIZE=1048576000  # 1GB
```

### Thay đổi port:
Backend: Thay đổi `API_PORT` trong `.env`
Frontend: Thay đổi trong `vite.config.js`

## Liên hệ hỗ trợ

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub hoặc liên hệ qua email.
