import React, { useState } from 'react';

function MessageInput({ onSend }) {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim()) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder="Nhập tin nhắn..."
        style={{ flex: 1, padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
        onKeyDown={e => e.key === 'Enter' && handleSend()}
      />
      <button onClick={handleSend} style={{ padding: '8px 16px', borderRadius: 4, background: '#1890ff', color: 'white', border: 'none' }}>
        Gửi
      </button>
    </div>
  );
}

export default MessageInput;
