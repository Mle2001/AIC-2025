import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChat } from '../../hook/useChat';
import ChatWebSocketClient from '../../services/websocket';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import Keyframe from './Keyframe';
import VideoPlayer from './VideoPlayer';

const ChatWindow = ({ userId = 'user1', onSessionChange }) => {
  // Sử dụng custom hook useChat để quản lý toàn bộ logic chat
  const [sessionId] = useState(() => uuidv4());
  const {
    messages,
    sendMessage,
    isLoading,
    error,
    connectionStatus,
    isTyping,
    loadHistory
  } = useChat(sessionId);
  const [keyframes, setKeyframes] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);
  const [selectedKeyframes, setSelectedKeyframes] = useState([]);
  // Refs
  const wsClientRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const wsClient = new ChatWebSocketClient(sessionId, userId);
    wsClientRef.current = wsClient;

    // Setup message handler
    const unsubscribeMessage = wsClient.onMessage((data) => {
      handleIncomingMessage(data);
    });

    // Setup status handler
    const unsubscribeStatus = wsClient.onStatus((status) => {
      setConnectionStatus(status);
      if (status === 'reconnecting') {
        setIsTyping(false);
      }
    });

    wsClient.connect();
    onSessionChange && onSessionChange(sessionId);

    return () => {
      unsubscribeMessage();
      unsubscribeStatus();
      wsClient.disconnect();
    };
  }, [sessionId, userId, onSessionChange]);

  // Auto-scroll tới bottom khi có message mới
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  /**
   * Handle incoming messages từ ConversationOrchestrator
   */
  const handleIncomingMessage = (data) => {
    const { type, content, mediaReferences: refs, processingTime } = data;

    if (type === 'user_message') {
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          type: 'user',
          content,
          timestamp: new Date(),
          isUser: true
        }
      ]);
      setIsTyping(true);
    } else if (type === 'bot_response') {
      setIsTyping(false);
      const botMessage = {
        id: uuidv4(),
        type: 'bot',
        content,
        timestamp: new Date(),
        isUser: false,
        processingTime: processingTime || null,
        mediaReferences: refs || []
      };
      setMessages(prev => [...prev, botMessage]);
      if (refs && refs.length > 0) {
        setMediaReferences(refs);
        // Nếu refs có keyframes, tự động cập nhật keyframes state
        const keyframeRefs = refs.filter(ref => ref.type === 'keyframe');
        if (keyframeRefs.length > 0) {
          setKeyframes(keyframeRefs.map(kf => ({
            ...kf,
            videoUrl: kf.video_url || kf.videoUrl || null,
            timestamp: kf.timestamp || 0
          })));
        }
        // Nếu refs có video, tự động chọn video đầu tiên
        const firstVideo = refs.find(ref => ref.type === 'video');
        if (firstVideo) setSelectedVideo(firstVideo.url || firstVideo.videoUrl);
      }
    } else if (type === 'error') {
      setIsTyping(false);
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          type: 'error',
          content,
          timestamp: new Date(),
          isUser: false
        }
      ]);
    }
  };

  // Gửi message qua WebSocket
  const handleSendMessage = (msg) => {
    if (wsClientRef.current) {
      wsClientRef.current.sendMessage(msg);
    }
  };

  // Hàm xử lý click vào keyframe
  const handleKeyframeClick = (keyframe) => {
    setSelectedVideo(keyframe.videoUrl);
    setSelectedTimestamp(keyframe.timestamp);
  };

  // Hàm xử lý chọn keyframe để export
  const handleSelectKeyframe = (keyframe) => {
    setSelectedKeyframes(prev => {
      const exists = prev.find(kf => kf.id === keyframe.id);
      if (exists) {
        return prev.filter(kf => kf.id !== keyframe.id);
      } else {
        if (prev.length >= 100) return prev; // Giới hạn 100
        return [...prev, keyframe];
      }
    });
  };

  // Export CSV
  const handleExportCSV = () => {
    const rows = [
      ['video_name', 'frame_index'],
      ...selectedKeyframes.slice(0, 100).map(kf => [kf.videoName || kf.video_name || '', kf.frameIndex || kf.frame_index || kf.timestamp || ''])
    ];
    const csvContent = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'keyframes_export.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Lấy danh sách các input user đã nhập
  const userInputs = messages.filter(m => m.type === 'user').map(m => m.content);

  // Auto-scroll helper
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div style={{ display: 'flex', gap: 24, height: '100%' }}>
      {/* Chat + Video */}
      <div style={{ flex: 2, minWidth: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ background: '#fff', borderRadius: 8, padding: 24, marginBottom: 24, boxShadow: '0 2px 8px #f0f1f2', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginTop: 0, marginBottom: 16 }}>💬 Chat Window</h3>
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <MessageList messages={messages} />
            {isTyping && (
              <div style={{ color: '#888', fontStyle: 'italic', margin: '8px 0 0 8px' }}>Bot is typing...</div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <MessageInput onSend={sendMessage} disabled={isLoading || connectionStatus !== 'connected'} />
          {error && <div style={{color:'red', marginTop:8}}>{error}</div>}
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
        <button onClick={handleExportCSV} disabled={selectedKeyframes.length === 0} style={{marginBottom:8}}>
          Export CSV ({selectedKeyframes.length}/100)
        </button>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          <ul style={{padding:0, margin:0, listStyle:'none'}}>
            {keyframes.map((kf, idx) => (
              <li key={kf.id || idx} style={{display:'flex',alignItems:'center',marginBottom:8}}>
                <input
                  type="checkbox"
                  checked={selectedKeyframes.some(sel => sel.id === kf.id)}
                  onChange={() => handleSelectKeyframe(kf)}
                  style={{marginRight:8}}
                  disabled={selectedKeyframes.length >= 100 && !selectedKeyframes.some(sel => sel.id === kf.id)}
                />
                <span
                  style={{cursor:'pointer',color:'#1890ff',textDecoration:'underline'}}
                  onClick={() => handleKeyframeClick(kf)}
                >
                  {kf.videoName || kf.video_name || 'Video'} - Frame {kf.frameIndex || kf.frame_index || kf.timestamp}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
      {/* Context Panel (hiển thị input user) */}
      <div style={{ flex: 1, minWidth: 0, background: '#fff', borderRadius: 8, padding: 24, boxShadow: '0 2px 8px #f0f1f2', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ marginTop: 0, marginBottom: 16 }}>📄 Context Panel</h3>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          <b>Conversation History:</b>
          <ul style={{ margin: '8px 0 16px 16px', color: '#444', fontSize: 14 }}>
            {userInputs.length === 0 && <li>Chưa có input nào</li>}
            {userInputs.map((input, idx) => (
              <li key={idx}>{input}</li>
            ))}
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
