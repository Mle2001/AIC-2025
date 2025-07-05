// src/components/Admin/VideoProcessManager.jsx
import React, { useState } from 'react';

// Dummy data for processing tasks
const initialTasks = [
  { id: 1, video: 'video1.mp4', status: 'Completed', progress: 100, result: 'Success' },
  { id: 2, video: 'video2.avi', status: 'Processing', progress: 60, result: null },
  { id: 3, video: 'video3.mov', status: 'Failed', progress: 100, result: 'Error: Format not supported' },
];

function VideoProcessManager() {
  const [tasks, setTasks] = useState(initialTasks);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [processing, setProcessing] = useState(false);

  // Dummy list of videos for selection
  const videoOptions = ['video1.mp4', 'video2.avi', 'video3.mov'];

  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedVideos(videoOptions);
    } else {
      setSelectedVideos([]);
    }
  };

  const handleSelectVideo = (e, name) => {
    if (e.target.checked) {
      setSelectedVideos(prev => [...prev, name]);
    } else {
      setSelectedVideos(prev => prev.filter(v => v !== name));
    }
  };

  const handleProcess = () => {
    if (!selectedVideos.length) return;
    setProcessing(true);
    // Simulate adding new processing tasks for all selected videos
    const newTasks = selectedVideos.map((video, idx) => ({
      id: tasks.length + idx + 1,
      video,
      status: 'Processing',
      progress: 0,
      result: null,
    }));
    setTasks(prev => [...newTasks, ...prev]);
    // Simulate progress for all
    let prog = 0;
    const interval = setInterval(() => {
      prog += 20;
      setTasks(current => current.map(t =>
        newTasks.some(nt => nt.id === t.id)
          ? { ...t, progress: Math.min(prog, 100), status: prog >= 100 ? 'Completed' : 'Processing', result: prog >= 100 ? 'Success' : null }
          : t
      ));
      if (prog >= 100) {
        clearInterval(interval);
        setProcessing(false);
        setSelectedVideos([]);
      }
    }, 500);
  };

  return (
    <div style={{ margin: '16px 0' }}>
      <h3>Process Video</h3>
      <div style={{ marginBottom: 16 }}>
        <label>
          <input
            type="checkbox"
            checked={selectedVideos.length === videoOptions.length}
            onChange={handleSelectAll}
            style={{ marginRight: 8 }}
          />
          Chọn tất cả video đã upload
        </label>
        <ul style={{ listStyle: 'none', padding: 0, margin: '8px 0 0 0', maxHeight: 120, overflowY: 'auto' }}>
          {videoOptions.map((video, idx) => (
            <li key={video}>
              <label>
                <input
                  type="checkbox"
                  checked={selectedVideos.includes(video)}
                  onChange={e => handleSelectVideo(e, video)}
                  style={{ marginRight: 8 }}
                />
                {video}
              </label>
            </li>
          ))}
        </ul>
        <button
          onClick={handleProcess}
          disabled={processing || !selectedVideos.length}
          style={{
            marginTop: 12,
            padding: '6px 16px',
            backgroundColor: processing || !selectedVideos.length ? '#d9d9d9' : '#52c41a',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: processing || !selectedVideos.length ? 'not-allowed' : 'pointer'
          }}
        >
          {processing ? 'Processing...' : 'Process Selected'}
        </button>
      </div>
      <h4>Danh sách tác vụ process</h4>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
        <thead>
          <tr style={{ background: '#fafafa' }}>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Video</th>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Trạng thái</th>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Tiến trình</th>
            <th style={{ padding: 8, border: '1px solid #eee' }}>Kết quả</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map(task => (
            <tr key={task.id}>
              <td style={{ padding: 8, border: '1px solid #eee' }}>{task.video}</td>
              <td style={{ padding: 8, border: '1px solid #eee' }}>{task.status}</td>
              <td style={{ padding: 8, border: '1px solid #eee', minWidth: 120 }}>
                <div style={{ width: '100%', background: '#f0f0f0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${task.progress}%`, height: '100%', background: task.status === 'Failed' ? '#ff4d4f' : '#1890ff', transition: 'width 0.3s' }} />
                </div>
                <span style={{ fontSize: 12 }}>{task.progress}%</span>
              </td>
              <td style={{ padding: 8, border: '1px solid #eee', color: task.status === 'Failed' ? '#ff4d4f' : '#52c41a' }}>{task.result || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default VideoProcessManager;
