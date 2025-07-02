import React from 'react';

function MessageList({ messages }) {
  return (
    <div style={{ minHeight: 120, maxHeight: 240, overflowY: 'auto', marginBottom: 8 }}>
      {messages.map((msg, idx) => (
        <div key={idx} style={{
          margin: '8px 0',
          color: msg.from === 'user' ? '#333' : '#1890ff',
          background: msg.from === 'user' ? '#f0f7ff' : '#e6f9ea',
          borderRadius: 6,
          padding: 12
        }}>
          <b>{msg.from === 'user' ? 'User' : 'AI'}:</b> {msg.text}
        </div>
      ))}
      <div style={{ color: '#888', fontStyle: 'italic', marginTop: 8 }}>[Typing...]</div>
    </div>
  );
}

export default MessageList;
