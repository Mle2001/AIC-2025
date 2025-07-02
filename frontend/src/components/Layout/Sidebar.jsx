import React from 'react';
import { Link, useLocation } from 'react-router-dom';

function Sidebar() {
  const location = useLocation();
  return (
    <div className="sidebar" style={{
      width: 250,
      height: '100vh',
      backgroundColor: '#001529',
      color: 'white',
      padding: '20px 0',
      position: 'fixed',
      left: 0,
      top: 0
    }}>
      <div className="logo" style={{ padding: '0 20px', marginBottom: 30 }}>
        <h2>AIC 2025</h2>
      </div>
      <div className="menu">
        <Link 
          to="/dashboard"
          className={`menu-item ${location.pathname === '/dashboard' ? 'active' : ''}`}
          style={{
            display: 'block',
            padding: '15px 20px',
            color: 'white',
            textDecoration: 'none',
            backgroundColor: location.pathname === '/dashboard' ? '#1890ff' : 'transparent'
          }}
        >
          Dashboard
        </Link>
        <Link 
          to="/chat"
          className={`menu-item ${location.pathname === '/chat' ? 'active' : ''}`}
          style={{
            display: 'block',
            padding: '15px 20px',
            color: 'white',
            textDecoration: 'none',
            backgroundColor: location.pathname === '/chat' ? '#1890ff' : 'transparent'
          }}
        >
          Chat
        </Link>
      </div>
    </div>
  );
}

export default Sidebar;
