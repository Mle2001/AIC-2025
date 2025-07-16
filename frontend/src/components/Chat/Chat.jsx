import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function Chat({ selectedVideo = null, showVideoSelector = false }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [videos, setVideos] = useState([]);
  const [currentVideo, setCurrentVideo] = useState(selectedVideo);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  // Scroll to bottom khi có message mới
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load videos nếu cần hiển thị video selector
  useEffect(() => {
    if (showVideoSelector) {
      loadVideos();
    }
  }, [showVideoSelector]);

  // Reset messages khi chọn video khác
  useEffect(() => {
    if (currentVideo) {
      setMessages([
        {
          id: 1,
          type: 'system',
          content: `🎬 Connected to video: ${currentVideo.original_name}`,
          timestamp: new Date()
        }
      ]);
    } else if (!showVideoSelector) {
      setMessages([]);
    }
  }, [currentVideo, showVideoSelector]);

  // Set video từ props
  useEffect(() => {
    setCurrentVideo(selectedVideo);
  }, [selectedVideo]);

  const loadVideos = async () => {
    try {
      const response = await axios.get('/api/chat/videos');
      setVideos(response.data.videos || []);
    } catch (error) {
      console.error('Error loading videos:', error);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError('');

    try {
      let response;
      
      if (currentVideo) {
        // Video chat mode
        response = await axios.post(`/api/upload/video/${currentVideo.file_id}/query`, {
          file_id: currentVideo.file_id,
          query: inputMessage
        });
      } else {
        // Regular chat mode
        response = await axios.post('/api/chat/message', {
          message: inputMessage,
          model: 'gpt-3.5-turbo'
        });
      }

      const aiMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
        metadata: response.data.metadata
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `Error: ${error.response?.data?.detail || 'Failed to send message'}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Suggested questions
  const suggestedQuestions = currentVideo ? [
    "Tóm tắt nội dung video này",
    "Những điểm chính trong video là gì?",
    "Có những ai xuất hiện trong video?",
    "Video này nói về chủ đề gì?",
    "Thời lượng video là bao lâu?"
  ] : [
    "Bạn có thể giúp tôi gì?",
    "Giải thích về AI và machine learning",
    "Tôi cần hỗ trợ với lập trình",
    "Tạo một kế hoạch học tập"
  ];

  const handleSuggestedQuestion = (question) => {
    setInputMessage(question);
  };

  const handleVideoSelect = (video) => {
    setCurrentVideo(video);
  };

  return (
    <div style={{ 
      marginLeft: showVideoSelector ? '270px' : '0px', 
      padding: '20px', 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column' 
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2>💬 {currentVideo ? `Chat with ${currentVideo.original_name}` : 'Chat'}</h2>
        
        {/* Video Selector */}
        {showVideoSelector && (
          <select 
            value={currentVideo?.file_id || ''} 
            onChange={(e) => {
              const video = videos.find(v => v.file_id === e.target.value);
              handleVideoSelect(video);
            }}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            <option value="">Select video...</option>
            {videos.map(video => (
              <option key={video.file_id} value={video.file_id}>
                {video.original_name}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', background: '#f5f5f5', borderRadius: '8px', marginBottom: '20px' }}>
        {messages.map((message) => (
          <div key={message.id} style={{ 
            marginBottom: '15px', 
            padding: '12px', 
            borderRadius: '8px',
            maxWidth: '80%',
            backgroundColor: message.type === 'user' ? '#007bff' : 
                           message.type === 'error' ? '#dc3545' : 
                           message.type === 'system' ? '#28a745' : '#fff',
            color: message.type === 'user' || message.type === 'system' ? 'white' : 'black',
            marginLeft: message.type === 'user' ? 'auto' : '0',
            marginRight: message.type === 'user' ? '0' : 'auto'
          }}>
            <div>
              <strong>
                {message.type === 'user' ? 'You' : 
                 message.type === 'error' ? 'Error' : 
                 message.type === 'system' ? 'System' : 'AI'}:
              </strong> {message.content}
            </div>
            <div style={{ fontSize: '0.8em', opacity: 0.8, marginTop: '5px' }}>
              {formatTime(message.timestamp)}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div style={{ padding: '12px', background: '#e9ecef', borderRadius: '8px', maxWidth: '80%' }}>
            <strong>AI:</strong> Thinking...
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length <= 1 && (
        <div style={{ marginBottom: '20px' }}>
          <h4>Suggested Questions:</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                style={{
                  padding: '8px 12px',
                  backgroundColor: '#f8f9fa',
                  border: '1px solid #dee2e6',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={currentVideo ? "Ask about the video..." : "Type your message..."}
          disabled={isLoading}
          style={{ flex: 1, padding: '12px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button 
          type="submit" 
          disabled={isLoading || !inputMessage.trim()}
          style={{ 
            padding: '12px 20px', 
            backgroundColor: '#007bff', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px', 
            cursor: isLoading ? 'not-allowed' : 'pointer' 
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>
          {error}
        </div>
      )}
    </div>
  );
}

export default Chat;