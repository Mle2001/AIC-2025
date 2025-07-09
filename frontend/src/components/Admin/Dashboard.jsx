// src/components/Admin/Dashboard.jsx
import React from 'react';
import VideoUpload from './VideoUpload';
import VideoProcessManager from './VideoProcessManager';

function Dashboard() {
  return (
    <div style={{ marginLeft: 270, padding: 32, minHeight: '100vh', background: '#f4f6f8' }}>
      <h2 style={{ marginBottom: 24 }}>Dashboard</h2>
      {/* Bố cục 2 cột: Upload bên trái, Process Video bên phải */}
      <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ flex: 2, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <h3>Quản lý video</h3>
          <VideoUpload />
        </div>
        <div style={{ flex: 1, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <VideoProcessManager />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;