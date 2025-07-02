import React, { useState } from 'react';
import ChatWindow from './ChatWindow';

function Chat() {
  // sessionId và các props khác có thể lấy từ context hoặc props nếu cần
  return (
    <div style={{
      marginLeft: 270, // giữ sidebar
      padding: 0,
      minHeight: '100vh',
      background: '#f4f6f8',
      boxSizing: 'border-box',
      width: 'calc(100vw - 270px)', // trừ sidebar
      height: '100vh',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'stretch',
    }}>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
        <ChatWindow />
      </div>
    </div>
  );
}

export default Chat;