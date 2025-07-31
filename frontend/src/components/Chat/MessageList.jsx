import React, { useEffect, useState } from 'react';

function MessageList({ messages, sessionId = 'default' }) {
  const [history, setHistory] = useState([]);

  const storageKey = `chat_history_${sessionId}`;

  // Lưu lịch sử vào localStorage khi messages thay đổi
  useEffect(() => {
    if (messages && messages.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(messages));
      setHistory(messages);
    }
  }, [messages, storageKey]);

  // Khi load lại trang, lấy lịch sử từ localStorage
  useEffect(() => {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, [storageKey]);

  // Xóa lịch sử theo session
  const handleClearHistory = () => {
    localStorage.removeItem(storageKey);
    setHistory([]);
  };

  return (
    <div style={{ minHeight: 120, maxHeight: 240, overflowY: 'auto', marginBottom: 8 }}>
      <button onClick={handleClearHistory} style={{marginBottom:8, float:'right'}}>Xóa lịch sử</button>
      {history.map((msg, idx) => (
        <div key={idx} style={{
          margin: '8px 0',
          color: msg.type === 'user' ? '#333' : '#1890ff',
          background: msg.type === 'user' ? '#f0f7ff' : '#e6f9ea',
          borderRadius: 6,
          padding: 12
        }}>
          <b>{msg.type === 'user' ? 'User' : 'AI'}:</b> {msg.content}
        </div>
      ))}
      <div style={{ color: '#888', fontStyle: 'italic', marginTop: 8 }}>[Typing...]</div>
    </div>
  );
}

export default MessageList;
