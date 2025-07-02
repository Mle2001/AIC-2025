// src/components/Admin/VideoUpload.jsx
import React, { useRef, useState } from 'react';

function VideoUpload() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const inputRef = useRef();

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) return;
    setUploading(true);
    setProgress(0);
    
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('videos', file));

    try {
      const xhr = new XMLHttpRequest();
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setProgress(Math.round(percentComplete));
        }
      });

      xhr.onload = () => {
        if (xhr.status === 200) {
          alert('Upload successful!');
          setSelectedFiles([]);
          inputRef.current.value = '';
        } else {
          alert('Upload failed!');
        }
        setUploading(false);
      };

      xhr.onerror = () => {
        alert('Upload failed!');
        setUploading(false);
      };

      xhr.open('POST', '/api/upload/videos');
      xhr.send(formData);
    } catch (error) {
      console.error('Upload error:', error);
      setUploading(false);
    }
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

      {uploading && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ 
            width: '100%', 
            height: 8, 
            backgroundColor: '#f0f0f0',
            borderRadius: 4,
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              backgroundColor: '#1890ff',
              transition: 'width 0.3s'
            }} />
          </div>
          <p>{progress}% Uploaded</p>
        </div>
      )}

      <button 
        onClick={handleUpload} 
        disabled={uploading || !selectedFiles.length}
        style={{
          padding: '8px 16px',
          backgroundColor: uploading || !selectedFiles.length ? '#d9d9d9' : '#1890ff',
          color: 'white',
          border: 'none',
          borderRadius: 4,
          cursor: uploading || !selectedFiles.length ? 'not-allowed' : 'pointer'
        }}
      >
        {uploading ? 'Uploading...' : 'Start Upload'}
      </button>
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