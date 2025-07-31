import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChat } from '../../hook/useChat';
import ChatWebSocketClient from '../../services/websocket';
import MessageList from './MessageList';

import MessageInput from './MessageInput';
import Result from './Result';

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
  // Refs

  const wsClientRef = useRef(null);
  const messagesEndRef = useRef(null);

  // State for results
  const [results, setResults] = useState([]);

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

  const handleIncomingMessage = (data) => {
    const { type, content, mediaReferences: refs, processingTime, results: resultList } = data;

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
      // Nếu có trường results từ backend, cập nhật danh sách kết quả
      if (Array.isArray(resultList)) {
        setResults(resultList);
      }
      if (refs && refs.length > 0) {
        setMediaReferences(refs);
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
            {/* Always show Result below messages */}
            <Result results={results} />
          </div>
          <MessageInput onSend={sendMessage} disabled={isLoading || connectionStatus !== 'connected'} />
          {error && <div style={{color:'red', marginTop:8}}>{error}</div>}
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
