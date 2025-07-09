// src/components/Admin/VideoProcessManager.jsx
import React, { useState } from 'react';
import { useVideo } from '../../hook/useVideo';

function VideoProcessManager() {
  // Hook quản lý video
  const { videos, processVideos, progress, results, isLoading, error, connectionStatus, reloadVideos } = useVideo();
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [processing, setProcessing] = useState(false);

  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedVideos(videos.map(v => v.id || v.name));
    } else {
      setSelectedVideos([]);
    }
  };

  const handleSelectVideo = (e, id) => {
    if (e.target.checked) {
      setSelectedVideos(prev => [...prev, id]);
    } else {
      setSelectedVideos(prev => prev.filter(v => v !== id));
    }
  };

  const handleProcess = () => {
    if (!selectedVideos.length) return;
    processVideos(selectedVideos);
    setProcessing(true);
  };

  return (
    <div style={{ margin: '16px 0' }}>
      <h3>Process Video</h3>
      <div style={{ marginBottom: 16 }}>
        <label>
          <input
            type="checkbox"
            checked={selectedVideos.length === videos.length}
            onChange={handleSelectAll}
            style={{ marginRight: 8 }}
          />
          Chọn tất cả video đã upload
        </label>
        <ul style={{ listStyle: 'none', padding: 0, margin: '8px 0 0 0', maxHeight: 120, overflowY: 'auto' }}>
          {videos.map((video, idx) => (
            <li key={video.id || video.name}>
              <label>
                <input
                  type="checkbox"
                  checked={selectedVideos.includes(video.id || video.name)}
                  onChange={e => handleSelectVideo(e, video.id || video.name)}
                  style={{ marginRight: 8 }}
                />
                {video.name || video.filename}
                {progress[video.id || video.name] !== undefined && (
                  <span style={{marginLeft:8}}>
                    {progress[video.id || video.name]}%
                  </span>
                )}
                {results[video.id || video.name] && (
                  <span style={{marginLeft:8, color:'green'}}>✔️ Done</span>
                )}
              </label>
            </li>
          ))}
        </ul>
        <button
          onClick={handleProcess}
          disabled={processing || !selectedVideos.length || isLoading}
          style={{
            marginTop: 12,
            padding: '6px 16px',
            backgroundColor: processing || !selectedVideos.length || isLoading ? '#d9d9d9' : '#52c41a',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: processing || !selectedVideos.length || isLoading ? 'not-allowed' : 'pointer'
          }}
        >
          {processing ? 'Processing...' : 'Process Selected'}
        </button>
        <button onClick={reloadVideos} style={{marginLeft:8}}>Reload</button>
        {error && <div style={{color:'red'}}>{error}</div>}
        <div style={{marginTop:16}}>
          <b>WebSocket status:</b> {connectionStatus}
        </div>
      </div>
      <h4>Danh sách tác vụ process</h4>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
        <thead>
          <tr style={{ background: '#fafafa' }}>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Video</th>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Tiến trình</th>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Kết quả</th>
          </tr>
        </thead>
        <tbody>
          {videos.map(video => (
            <tr key={video.id || video.name}>
              <td style={{ padding: 8, border: '1px solid #eee' }}>{video.name || video.filename}</td>
              <td style={{ padding: 8, border: '1px solid #eee', minWidth: 120 }}>
                <div style={{ width: '100%', background: '#f0f0f0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${progress[video.id || video.name] || 0}%`, height: '100%', background: '#1890ff', transition: 'width 0.3s' }} />
                </div>
                <span style={{ fontSize: 12 }}>{progress[video.id || video.name] || 0}%</span>
              </td>
              <td style={{ padding: 8, border: '1px solid #eee', color: results[video.id || video.name] ? '#52c41a' : '#888' }}>{results[video.id || video.name] || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default VideoProcessManager;
