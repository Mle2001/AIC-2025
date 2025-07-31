import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
// import { Chat } from '../Chat';

const AdminDashboard = () => {
  const [videos, setVideos] = useState([]);
  // const [selectedVideo, setSelectedVideo] = useState(null);
  const [refreshVideoTrigger, setRefreshVideoTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState('manage');
  const [processing, setProcessing] = useState(false);
  const [processJobs, setProcessJobs] = useState([]);
  const [processError, setProcessError] = useState('');
  // Fetch process jobs
  const fetchProcessJobs = async () => {
    try {
      const response = await axios.get('/api/chat/process-jobs');
      setProcessJobs(response.data.jobs || []);
    } catch (error) {
      setProcessJobs([]);
    }
  };

  useEffect(() => {
    fetchProcessJobs();
    // Optionally poll for updates
    const interval = setInterval(fetchProcessJobs, 5000);
    return () => clearInterval(interval);
  }, []);
  // Process videos handler
  const handleProcessVideos = async () => {
    setProcessing(true);
    setProcessError('');
    try {
      await axios.post('/api/chat/process-videos');
      fetchProcessJobs();
    } catch (err) {
      setProcessError('Failed to start processing.');
    } finally {
      setProcessing(false);
    }
  };

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


  // Upload video handler
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    setUploading(true);
    setUploadError('');
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append('video', file);
        await axios.post('/api/chat/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }
      setRefreshVideoTrigger(prev => prev + 1);
    } catch (err) {
      setUploadError('Upload failed.');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div style={{
      height: '100vh',
      marginLeft: 240,
      marginRight: 10,
      display: 'flex',
      flexDirection: 'column',
      background: '#f0f2f5',
      width: 'calc(100vw - 250px)',
      minWidth: 0,
      boxSizing: 'border-box',
      overflow: 'hidden',
      position: 'relative'
    }}>
      <div style={{ 
        background: '#fff',
        padding: '16px 24px',
        borderBottom: '1px solid #e8e8e8',
        display: 'flex',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, color: '#1890ff' }}>
          🎬 Admin Dashboard
        </h1>
      </div>
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'flex-start',
        justifyContent: 'stretch',
        padding: '40px 40px 0 40px',
        minWidth: 0,
        width: '100%',
        gap: 40,
        overflow: 'auto',
        height: 'calc(100vh - 80px)'
      }}>
        {/* Left: Upload & Uploaded Videos */}
        <div style={{ flex: 1, minWidth: 300, maxWidth: '50%', flexBasis: 0, display: 'flex', flexDirection: 'column', gap: 32, height: '100%' }}>
          {/* Upload Video */}
          <div style={{ background: '#fff', padding: 24, borderRadius: 8, boxShadow: '0 2px 8px #0001' }}>
            <h2 style={{ marginBottom: 16 }}>Upload Video</h2>
            <input
              type="file"
              accept="video/*"
              ref={fileInputRef}
              onChange={handleFileChange}
              disabled={uploading}
              multiple
              style={{ marginBottom: 12 }}
            />
            {uploadError && <div style={{ color: 'red', marginTop: 10 }}>{uploadError}</div>}
          </div>
          {/* Uploaded Videos */}
          <div style={{ background: '#fff', padding: 24, borderRadius: 8, boxShadow: '0 2px 8px #0001' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <h2 style={{ margin: 0 }}>Uploaded Videos</h2>
              <button
                onClick={handleProcessVideos}
                disabled={processing || videos.length === 0}
                style={{ padding: '8px 20px', background: '#52c41a', color: '#fff', border: 'none', borderRadius: 4, cursor: processing || videos.length === 0 ? 'not-allowed' : 'pointer' }}
              >
                {processing ? 'Processing...' : 'Process'}
              </button>
            </div>
            {processError && <div style={{ color: 'red', marginBottom: 10 }}>{processError}</div>}
            {videos.length === 0 ? (
              <div style={{ color: '#999' }}>No videos uploaded yet.</div>
            ) : (
              <ul style={{ listStyle: 'none', padding: 0 }}>
                {videos.map((video) => (
                  <li key={video.file_id} style={{ padding: '12px 0', borderBottom: '1px solid #eee', display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <strong>{video.original_name}</strong>
                      <div style={{ fontSize: 12, color: '#888' }}>Status: {video.status}</div>
                      <div style={{ fontSize: 11, color: '#bbb' }}>{new Date(video.created_at).toLocaleString()}</div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        {/* Right: Processing Jobs */}
        <div style={{ flex: 1, minWidth: 300, maxWidth: '40%', flexBasis: 0, background: '#fff', padding: 24, borderRadius: 8, boxShadow: '0 2px 8px #0001', minHeight: 400, height: '100%' }}>
          <h2 style={{ marginBottom: 16 }}>Processing Jobs</h2>
          {processJobs.length === 0 ? (
            <div style={{ color: '#999' }}>No processing jobs found.</div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {processJobs.map((job) => (
                <li key={job.id} style={{ padding: '12px 0', borderBottom: '1px solid #eee', display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <strong>Job #{job.id}</strong>
                    <div style={{ fontSize: 12, color: '#888' }}>Status: {job.status}</div>
                    <div style={{ fontSize: 11, color: '#bbb' }}>{new Date(job.created_at).toLocaleString()}</div>
                  </div>
                  <div style={{ fontSize: 12, color: '#1890ff' }}>{job.message || ''}</div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
