import React, { useState } from 'react';
import { VideoUpload, VideoList } from './';
import './VideoRAG.css';

const VideoRAGDashboard = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState('upload');

  // Callback khi video được upload
  const handleVideoUploaded = (videoData) => {
    setRefreshTrigger(prev => prev + 1);
    if (videoData.message?.includes('processed')) {
      // Nếu video đã được process, chuyển sang tab list
      setActiveTab('list');
    } else {
      // Nếu chưa process, chuyển sang tab list
      setActiveTab('list');
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <VideoUpload 
            onVideoUploaded={handleVideoUploaded} 
          />
        );
      case 'list':
        return (
          <VideoList 
            refreshTrigger={refreshTrigger}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div style={{ marginLeft: '270px', padding: '20px', minHeight: '100vh' }}>
      <div className="videorag-dashboard">
        <div className="dashboard-header">
          <h1>🎬 VideoRAG Dashboard</h1>
          <p>Upload videos and chat with them using AI</p>
        </div>

        {/* Navigation Tabs */}
        <div className="dashboard-tabs">
          <button 
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📤 Upload Video
          </button>
          <button 
            className={`tab-btn ${activeTab === 'list' ? 'active' : ''}`}
            onClick={() => setActiveTab('list')}
          >
            📋 My Videos
          </button>
        </div>

        {/* Tab Content */}
        <div className="dashboard-content">
          {renderTabContent()}
        </div>

        {/* Footer */}
        <div className="dashboard-footer">
          <p>
            🚀 Powered by VideoRAG - Upload videos and interact with them using AI
          </p>
          <p>
            💬 Use the Chat section to interact with your processed videos
          </p>
        </div>
      </div>
    </div>
  );
};

export default VideoRAGDashboard;
