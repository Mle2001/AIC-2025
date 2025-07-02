// src/components/Admin/Dashboard.jsx
import React, { useState } from 'react';
import VideoUpload from './VideoUpload';
import Analytics from './Analytics';

// Dummy data for uploaded videos
const initialVideos = [
  { name: 'video1.mp4', size: 10485760, status: 'Processed' },
  { name: 'video2.avi', size: 5242880, status: 'Processing' },
  { name: 'video3.mov', size: 2097152, status: 'Failed' },
];

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function Dashboard() {
  const [videos, setVideos] = useState(initialVideos);

  // Optionally, handle new uploads to update the list
  // const handleUploadSuccess = (newVideos) => setVideos([...videos, ...newVideos]);

  return (
    <div style={{ marginLeft: 270, padding: 32, minHeight: '100vh', background: '#f4f6f8' }}>
      <h2 style={{ marginBottom: 24 }}>Dashboard</h2>
      <div style={{ background: '#fff', borderRadius: 8, padding: 24, marginBottom: 32, boxShadow: '0 2px 8px #f0f1f2' }}>
        <h3>Quản lý video đã upload</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
          <thead>
            <tr style={{ background: '#fafafa' }}>
              <th style={{ padding: 8, border: '1px solid #eee' }}>Tên video</th>
              <th style={{ padding: 8, border: '1px solid #eee' }}>Dung lượng</th>
              <th style={{ padding: 8, border: '1px solid #eee' }}>Trạng thái</th>
            </tr>
          </thead>
          <tbody>
            {videos.map((video, idx) => (
              <tr key={idx}>
                <td style={{ padding: 8, border: '1px solid #eee' }}>{video.name}</td>
                <td style={{ padding: 8, border: '1px solid #eee' }}>{formatFileSize(video.size)}</td>
                <td style={{ padding: 8, border: '1px solid #eee' }}>{video.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <VideoUpload />
        </div>
        <div style={{ flex: 1, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <Analytics />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;