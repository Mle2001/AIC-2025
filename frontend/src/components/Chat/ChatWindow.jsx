import React, { useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import Keyframe from './Keyframe';
import VideoPlayer from './VideoPlayer';

function ChatWindow() {
  const [messages, setMessages] = useState([
    { from: 'user', text: 'Tìm cho tôi video về nấu phở có người thêm rau húng quế' },
    { from: 'ai', text: 'Tôi tìm thấy 3 video về nấu phở có cảnh thêm rau húng quế. Video đầu tiên có hướng dẫn chi tiết nhất...' }
  ]);
  const [keyframes, setKeyframes] = useState([
    { id: 1, label: 'Keyframe 1', time: '00:12:34', confidence: 0.95, videoUrl: '', timestamp: 754 },
    { id: 2, label: 'Keyframe 2', time: '00:23:45', confidence: 0.92, videoUrl: '', timestamp: 1425 }
  ]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);

  const handleSendMessage = (msg) => {
    setMessages([...messages, { from: 'user', text: msg }]);
    // TODO: Gửi message tới backend và nhận kết quả keyframe, video
  };

  const handleKeyframeClick = (keyframe) => {
    setSelectedVideo(keyframe.videoUrl);
    setSelectedTimestamp(keyframe.timestamp);
  };

  return (
    <div style={{ display: 'flex', gap: 24, height: '100%' }}>
      {/* Chat + Video */}
      <div style={{ flex: 2, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ background: '#fff', borderRadius: 8, padding: 24, marginBottom: 24, boxShadow: '0 2px 8px #f0f1f2', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginTop: 0, marginBottom: 16 }}>💬 Chat Window</h3>
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <MessageList messages={messages} />
          </div>
          <MessageInput onSend={handleSendMessage} />
        </div>
        <div style={{ background: '#181c24', borderRadius: 8, padding: 24, color: '#fff', boxShadow: '0 2px 8px #f0f1f2', marginBottom: 24 }}>
          <h3 style={{ marginTop: 0, marginBottom: 16 }}>🎬 Video Player</h3>
          <VideoPlayer videoUrl={selectedVideo} timestamp={selectedTimestamp} />
          <div style={{ marginTop: 16 }}>
            <b>Timeline:</b> <span style={{ background: '#fff', color: '#181c24', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>Highlighted Segments</span>
          </div>
        </div>
      </div>
      {/* Keyframe List */}
      <div style={{ flex: 1, minWidth: 0, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ marginTop: 0, marginBottom: 16 }}>🔍 Search Results</h3>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          <Keyframe keyframes={keyframes} onSelect={handleKeyframeClick} />
        </div>
      </div>
      {/* Context Panel (optional) */}
      <div style={{ flex: 1, minWidth: 0, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ marginTop: 0, marginBottom: 16 }}>📄 Context Panel</h3>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          <b>Conversation History:</b>
          <ul style={{ margin: '8px 0 16px 16px', color: '#444', fontSize: 14 }}>
            <li>Previous queries</li>
            <li>User context</li>
            <li>Preferences</li>
          </ul>
          <b>Suggested Questions:</b>
          <ul style={{ margin: '8px 0 0 16px', color: '#444', fontSize: 14 }}>
            <li>"Video nào chi tiết nhất?"</li>
            <li>"Có bao nhiêu bước?"</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ChatWindow;
