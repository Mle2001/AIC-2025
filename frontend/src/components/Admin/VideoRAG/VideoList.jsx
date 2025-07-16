import React, { useState, useEffect } from 'react';
import axios from 'axios';

const VideoList = ({ onVideoSelected, refreshTrigger, videos: propVideos }) => {
  const [videos, setVideos] = useState(propVideos || []);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Fetch videos từ API
  const fetchVideos = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/upload/videos');
      setVideos(response.data.videos);
      setError('');
    } catch (error) {
      console.error('Error fetching videos:', error);
      setError('Failed to load videos');
    } finally {
      setLoading(false);
    }
  };

  // Load videos khi component mount và khi có refreshTrigger
  useEffect(() => {
    if (propVideos) {
      setVideos(propVideos);
      setLoading(false);
    } else {
      fetchVideos();
    }
  }, [refreshTrigger, propVideos]);

  // Xóa video
  const handleDeleteVideo = async (fileId) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa video này?')) {
      return;
    }

    try {
      await axios.delete(`/api/upload/video/${fileId}`);
      // Refresh danh sách
      fetchVideos();
    } catch (error) {
      console.error('Error deleting video:', error);
      alert('Failed to delete video');
    }
  };

  // Process video nếu chưa được process
  const handleProcessVideo = async (fileId) => {
    try {
      await axios.post(`/api/upload/video/${fileId}/process`, {
        file_id: fileId
        // openai_api_key is optional, backend will use default if not provided
      });
      
      // Refresh danh sách
      fetchVideos();
      alert('Video processed successfully!');
    } catch (error) {
      console.error('Error processing video:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to process video';
      alert(`Processing failed: ${errorMessage}`);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'processed': return '#4CAF50';
      case 'uploaded': return '#FF9800';
      case 'error': return '#F44336';
      default: return '#757575';
    }
  };

  if (loading) {
    return <div className="loading">Loading videos...</div>;
  }

  return (
    <div className="video-list-container">
      <h3>Your Videos ({videos.length})</h3>
      
      {error && (
        <div className="error-message">
          <p>❌ {error}</p>
        </div>
      )}

      {videos.length === 0 ? (
        <div className="no-videos">
          <p>No videos uploaded yet. Upload your first video!</p>
        </div>
      ) : (
        <div className="video-grid">
          {videos.map((video) => (
            <div key={video.file_id} className="video-card">
              <div className="video-info">
                <h4>{video.original_name}</h4>
                <p className="file-size">{formatFileSize(video.file_size)}</p>
                <div className="status-badge">
                  <span 
                    className="status-dot" 
                    style={{ backgroundColor: getStatusColor(video.status) }}
                  />
                  <span className="status-text">{video.status}</span>
                </div>
                {video.error && (
                  <p className="error-text">Error: {video.error}</p>
                )}
              </div>

              <div className="video-actions">
                {video.processed && (
                  <button 
                    className="btn btn-primary"
                    onClick={() => onVideoSelected(video)}
                  >
                    💬 Chat with Video
                  </button>
                )}
                
                {!video.processed && video.status === 'uploaded' && (
                  <button 
                    className="btn btn-secondary"
                    onClick={() => {
                      handleProcessVideo(video.file_id);
                    }}
                  >
                    🔄 Process Video
                  </button>
                )}

                <button 
                  className="btn btn-danger"
                  onClick={() => handleDeleteVideo(video.file_id)}
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <button 
        className="btn btn-outline refresh-btn"
        onClick={fetchVideos}
      >
        🔄 Refresh
      </button>
    </div>
  );
};

export default VideoList;
