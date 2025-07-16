import React, { useState } from 'react';
import axios from 'axios';

const VideoUpload = ({ onVideoUploaded }) => {
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Kiểm tra file type
    if (!file.type.startsWith('video/')) {
      setError('Chỉ hỗ trợ file video');
      return;
    }

    // Kiểm tra file size (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File quá lớn (tối đa 500MB)');
      return;
    }

    setUploading(true);
    setError('');
    setSuccess('');
    setUploadProgress(0);

    try {
      // Tạo FormData
      const formData = new FormData();
      formData.append('file', file);
      formData.append('auto_process', 'true');

      // Upload video
      const response = await axios.post('/api/upload/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      setSuccess('Video uploaded successfully!');
      
      // Set processing state
      if (response.data.message?.includes('processed')) {
        setSuccess('Video uploaded and processed successfully!');
      } else {
        setProcessing(true);
        // Có thể thêm logic để poll processing status
      }

      // Callback để parent component biết video đã upload
      if (onVideoUploaded) {
        onVideoUploaded(response.data);
      }

    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
      setProcessing(false);
    }
  };

  return (
    <div className="video-upload-container">
      <h3>Upload Video</h3>

      <div className="upload-section">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileUpload}
          disabled={uploading || processing}
          className="file-input"
        />
        
        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p>Uploading: {uploadProgress}%</p>
          </div>
        )}

        {processing && (
          <div className="processing-status">
            <p>🔄 Processing video with VideoRAG...</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <p>❌ {error}</p>
          </div>
        )}

        {success && (
          <div className="success-message">
            <p>✅ {success}</p>
          </div>
        )}
      </div>

      <div className="upload-info">
        <h4>Hướng dẫn:</h4>
        <ul>
          <li>Chọn file video (tối đa 500MB)</li>
          <li>Video sẽ được index bằng VideoRAG</li>
          <li>Sau khi xử lý, bạn có thể chat với video</li>
        </ul>
      </div>
    </div>
  );
};

export default VideoUpload;
