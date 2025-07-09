// src/components/Admin/VideoUpload.jsx
import React, { useRef, useState } from 'react';
import { useVideo } from '../../hook/useVideo';

function VideoUpload() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const inputRef = useRef();
  // Hook quản lý video
  const { uploadVideo, videos, isLoading, error, reloadVideos } = useVideo();

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) return;
    // Upload từng file qua useVideo
    selectedFiles.forEach(file => uploadVideo(file));
    setSelectedFiles([]);
    inputRef.current.value = '';
  };

  return (
    <div style={{ margin: '16px 0' }}>
      <h3>Upload Videos</h3>
      
      <div style={{ 
        border: '2px dashed #d9d9d9', 
        padding: 20,
        borderRadius: 8,
        textAlign: 'center',
        marginBottom: 16
      }}>
        <input
          type="file"
          accept="video/*"
          multiple
          ref={inputRef}
          onChange={handleFileChange}
          style={{ marginBottom: 8 }}
        />
        <p style={{ color: '#666' }}>
          Support for all video formats. Max file size: 500MB
        </p>
      </div>

      {selectedFiles.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <h4>Selected Files:</h4>
          <ul style={{ maxHeight: 200, overflowY: 'auto' }}>
            {selectedFiles.map((file, idx) => (
              <li key={idx} style={{ marginBottom: 8 }}>
                {file.name} ({formatFileSize(file.size)})
              </li>
            ))}
          </ul>
        </div>
      )}

      {isLoading && <div>Đang tải danh sách video...</div>}
      {error && <div style={{color:'red'}}>{error}</div>}
      <button 
        onClick={handleUpload} 
        disabled={isLoading || !selectedFiles.length}
        style={{
          padding: '8px 16px',
          backgroundColor: isLoading || !selectedFiles.length ? '#d9d9d9' : '#1890ff',
          color: 'white',
          border: 'none',
          borderRadius: 4,
          cursor: isLoading || !selectedFiles.length ? 'not-allowed' : 'pointer'
        }}
      >
        {isLoading ? 'Uploading...' : 'Start Upload'}
      </button>

      <h4 style={{marginTop:24}}>Danh sách video đã upload:</h4>
      <ul>
        {videos && videos.length === 0 && <li>Chưa có video nào</li>}
        {videos && videos.map(v => <li key={v.id || v.name}>{v.name || v.filename}</li>)}
      </ul>
      <button onClick={reloadVideos} style={{marginTop:8}}>Reload</button>
    </div>
  );
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export default VideoUpload;