import React, { useState } from 'react';
import axios from 'axios';

const SimpleVideoUpload = ({ onVideoUploaded }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Only video files are allowed');
      return;
    }

    // Validate file size (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large (max 500MB)');
      return;
    }

    setUploading(true);
    setError('');
    setSuccess('');
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('auto_process', 'true'); // Auto process with backend API key

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
      
      if (onVideoUploaded) {
        onVideoUploaded(response.data);
      }

    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ 
      background: '#fff',
      padding: '20px',
      borderRadius: '8px',
      border: '1px solid #e8e8e8'
    }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>📤 Upload Video</h3>
      
      <div style={{ 
        border: '2px dashed #d9d9d9',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center',
        marginBottom: '16px'
      }}>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileUpload}
          disabled={uploading}
          style={{ 
            width: '100%',
            padding: '8px',
            border: '1px solid #d9d9d9',
            borderRadius: '4px'
          }}
        />
        
        {uploading && (
          <div style={{ marginTop: '12px' }}>
            <div style={{ 
              width: '100%',
              height: '8px',
              background: '#f0f0f0',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{ 
                width: `${uploadProgress}%`,
                height: '100%',
                background: '#1890ff',
                transition: 'width 0.3s ease'
              }} />
            </div>
            <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
              Uploading: {uploadProgress}%
            </p>
          </div>
        )}
      </div>

      {error && (
        <div style={{ 
          background: '#fff2f0',
          border: '1px solid #ffccc7',
          borderRadius: '4px',
          padding: '8px 12px',
          color: '#a8071a',
          fontSize: '14px'
        }}>
          ❌ {error}
        </div>
      )}

      {success && (
        <div style={{ 
          background: '#f6ffed',
          border: '1px solid #b7eb8f',
          borderRadius: '4px',
          padding: '8px 12px',
          color: '#389e0d',
          fontSize: '14px'
        }}>
          ✅ {success}
        </div>
      )}

      <div style={{ 
        fontSize: '12px',
        color: '#666',
        marginTop: '12px'
      }}>
        <p style={{ margin: '4px 0' }}>• Max file size: 500MB</p>
        <p style={{ margin: '4px 0' }}>• Supported formats: MP4, AVI, MOV, etc.</p>
        <p style={{ margin: '4px 0' }}>• Auto-processed with VideoRAG</p>
      </div>
    </div>
  );
};

export default SimpleVideoUpload;
