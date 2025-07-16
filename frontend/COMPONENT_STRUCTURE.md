# Cấu trúc thư mục Frontend Components

## Tổ chức theo chức năng

### 📁 Admin/
- **AdminDashboard.jsx** - Dashboard chính cho admin
- **Analytics.jsx** - Thống kê và phân tích
- **Dashboard.jsx** - Dashboard tổng quan
- **VideoProcessManager.jsx** - Quản lý xử lý video

#### 📁 Admin/VideoRAG/
- **VideoUpload.jsx** - Upload video (chức năng admin)
- **VideoList.jsx** - Danh sách video (quản lý admin)
- **VideoRAGDashboard.jsx** - Dashboard VideoRAG
- **SimpleVideoUpload.jsx** - Upload video đơn giản
- **VideoRAG.css** - Styling cho VideoRAG
- **index.js** - Export tất cả components

### 📁 Chat/
- **Chat.jsx** - Unified chat interface (supports both regular chat and video chat)
- **ChatWindow.jsx** - Cửa sổ chat
- **Keyframe.jsx** - Hiển thị keyframes
- **MessageInput.jsx** - Input tin nhắn
- **MessageList.jsx** - Danh sách tin nhắn
- **VideoPlayer.jsx** - Player video
- **index.js** - Export tất cả components

### 📁 Layout/
- **Sidebar.jsx** - Navigation sidebar

## Import Examples

```jsx
// Import from Admin VideoRAG
import { VideoUpload, VideoList, VideoRAGDashboard } from './components/Admin/VideoRAG';

// Import unified Chat (handles both regular and video chat)
import { Chat } from './components/Chat';

// Import layout
import Sidebar from './components/Layout/Sidebar';
```

## Lý do tổ chức này

1. **Phân chia rõ ràng theo chức năng**: Admin vs Chat vs Layout
2. **VideoRAG components được tổ chức theo context sử dụng**:
   - Upload/List/Dashboard → Admin functions
   - ~~VideoChat~~ → **Integrated into Chat component**
3. **Unified Chat Experience**: Một component Chat duy nhất xử lý cả regular chat và video chat
4. **Dễ maintain và scale**: Mỗi folder có responsibility rõ ràng
5. **Import paths logic**: Phản ánh đúng chức năng sử dụng
6. **API key management**: Tất cả được handle ở backend, không có input fields

## Chat Component Features

- **Regular Chat**: Chat thông thường với OpenAI API
- **Video Chat**: Chat với video đã được processed
- **Video Selector**: Dropdown để chọn video khi `showVideoSelector={true}`
- **Auto-detection**: Tự động detect video mode khi có `selectedVideo` prop
- **Unified Interface**: Một UI duy nhất cho cả hai chức năng
