import React from 'react';

const ShowVideo = ({ videoUrl, time = '00:00:00', title = '', onClose }) => {
  if (!videoUrl) return null;
  return (
    <div
      style={{
        background: '#f8fbff',
        borderRadius: 12,
        boxShadow: '0 2px 8px #e0e4ea',
        padding: 0,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        position: 'relative',
      }}
    >
      {/* Tiêu đề và nút đóng */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 22 }}>🎬</span>
        <span style={{ fontWeight: 600, fontSize: 18, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>{title || 'Video Player'}</span>
        <button
          onClick={onClose}
          style={{
            marginLeft: 8,
            background: 'transparent',
            border: 'none',
            fontSize: 22,
            cursor: 'pointer',
            color: '#888',
          }}
        >
          ×
        </button>
      </div>
      {/* Video player */}
      <div
        style={{
          background: '#181d27',
          borderRadius: 10,
          minHeight: 120,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: 12,
          position: 'relative',
          width: '100%',
          height: 180,
          overflow: 'hidden',
        }}
      >
        <video
          src={videoUrl + `#t=${time}`}
          controls
          autoPlay
          style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 8, background: '#181d27' }}
        />
      </div>
    </div>
  );
};

export default ShowVideo;

