// src/components/Admin/Dashboard.jsx
import React from 'react';
import { Link } from 'react-router-dom';

function Dashboard() {
  return (
    <div style={{ marginLeft: 270, padding: 32, minHeight: '100vh', background: '#f4f6f8' }}>
      <h2 style={{ marginBottom: 24 }}>Dashboard</h2>
      <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ flex: 2, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <h3>🎬 VideoRAG System</h3>
          <p>Upload videos and chat with them using AI-powered VideoRAG technology.</p>
          <Link 
            to="/videorag" 
            style={{ 
              display: 'inline-block', 
              padding: '12px 24px', 
              backgroundColor: '#1890ff', 
              color: 'white', 
              textDecoration: 'none', 
              borderRadius: '4px',
              marginTop: '16px'
            }}
          >
            🚀 Go to VideoRAG
          </Link>
        </div>
        <div style={{ flex: 1, minWidth: 320, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2' }}>
          <h3>💬 Chat System</h3>
          <p>Traditional chat interface with AI support.</p>
          <Link 
            to="/chat" 
            style={{ 
              display: 'inline-block', 
              padding: '12px 24px', 
              backgroundColor: '#52c41a', 
              color: 'white', 
              textDecoration: 'none', 
              borderRadius: '4px',
              marginTop: '16px'
            }}
          >
            💬 Go to Chat
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;