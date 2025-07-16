import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { VideoUpload, VideoList } from './VideoRAG';
import { Chat } from '../Chat';
import './VideoRAG/VideoRAG.css';

const AdminDashboard = () => {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [refreshVideoTrigger, setRefreshVideoTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState('chat');

  // Fetch videos
  const fetchVideos = async () => {
    try {
      const response = await axios.get('/api/chat/videos');
      setVideos(response.data.videos || []);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  useEffect(() => {
    fetchVideos();
  }, [refreshVideoTrigger]);

  const handleVideoSelected = (video) => {
    setSelectedVideo(video);
    setActiveTab('chat');
  };

  const handleVideoUploaded = () => {
    setRefreshVideoTrigger(prev => prev + 1);
  };

  return (
    <div style={{ 
      marginLeft: '250px',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: '#f0f2f5'
    }}>
      {/* Header */}
      <div style={{ 
        background: '#fff',
        padding: '16px 24px',
        borderBottom: '1px solid #e8e8e8',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, color: '#1890ff' }}>
          🎬 Admin Dashboard - VideoRAG Chat
        </h1>
        <div>
          <button 
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
            style={{ 
              padding: '8px 16px', 
              marginRight: '8px',
              border: 'none',
              background: activeTab === 'chat' ? '#1890ff' : '#f5f5f5',
              color: activeTab === 'chat' ? 'white' : '#666',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            💬 Chat
          </button>
          <button 
            className={`tab-btn ${activeTab === 'manage' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage')}
            style={{ 
              padding: '8px 16px',
              border: 'none',
              background: activeTab === 'manage' ? '#1890ff' : '#f5f5f5',
              color: activeTab === 'manage' ? 'white' : '#666',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            📋 Manage Videos
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left Panel - Content */}
        <div style={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          borderRight: '1px solid #e8e8e8'
        }}>
          {activeTab === 'chat' && (
            <div style={{ flex: 1, marginLeft: '-250px' }}>
              <Chat 
                selectedVideo={selectedVideo} 
                showVideoSelector={!selectedVideo}
              />
            </div>
          )}
          {activeTab === 'manage' && (
            <div style={{ padding: '20px', flex: 1, overflowY: 'auto' }}>
              <h3>📋 Video Management</h3>
              <div style={{ marginBottom: '30px' }}>
                <VideoUpload onVideoUploaded={handleVideoUploaded} />
              </div>
              <VideoList 
                videos={videos}
                onVideoSelected={handleVideoSelected}
                refreshTrigger={refreshVideoTrigger}
              />
            </div>
          )}
        </div>

        {/* Right Panel - Video List */}
        <div style={{ 
          width: '300px', 
          background: '#fff',
          borderLeft: '1px solid #e8e8e8',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ 
            padding: '16px',
            borderBottom: '1px solid #e8e8e8',
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span>📹</span>
            <span>My Videos</span>
          </div>
          
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {videos.length === 0 ? (
              <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
                No videos uploaded yet
              </div>
            ) : (
              videos.map((video) => (
                <div
                  key={video.file_id}
                  onClick={() => handleVideoSelected(video)}
                  style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid #f0f0f0',
                    cursor: 'pointer',
                    backgroundColor: selectedVideo?.file_id === video.file_id ? '#e6f7ff' : 'transparent',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedVideo?.file_id !== video.file_id) {
                      e.target.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedVideo?.file_id !== video.file_id) {
                      e.target.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {video.original_name}
                  </div>
                  <div style={{ 
                    fontSize: '12px', 
                    color: video.status === 'processed' ? '#52c41a' : '#faad14'
                  }}>
                    Status: {video.status}
                  </div>
                  <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                    {new Date(video.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
