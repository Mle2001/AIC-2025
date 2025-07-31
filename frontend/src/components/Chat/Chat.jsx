import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import MessageList from './MessageList';
import Result from './Result';
import ShowVideo from './ShowVideo';

function Chat({ selectedVideo = null, showVideoSelector = false }) {
  // State for checked results
  const [checkedResults, setCheckedResults] = useState([]);
  // Mock result data for demo layout
  const [results, setResults] = useState([
    {
      key: 1,
      title: 'L01_V015 24033',
      time: '00:12:34',
      confidence: '95%',
      frameImg: 'https://peach.blender.org/wp-content/uploads/title_anouncement.jpg?x11217',
      videoUrl: 'https://www.w3schools.com/html/mov_bbb.mp4'
    },
    {
      key: 2,
      title: 'L11_V018 21559',
      time: '00:23:45',
      confidence: '92%',
      frameImg: 'https://www.w3schools.com/html/pic_trulli.jpg',
      videoUrl: 'https://www.w3schools.com/html/movie.mp4'
    }
  ]);

  // State for ShowVideo
  const [showVideo, setShowVideo] = useState(null); // { videoUrl, time, title }
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

    // MOCK: Trả về tin nhắn mẫu thay vì gọi API
    setTimeout(() => {
      const aiMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: `🤖 Đây là phản hồi mẫu cho: "${inputMessage}"`,
        timestamp: new Date(),
        metadata: {}
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 700);
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

  // Handle check/uncheck result
  const handleCheckResult = (key) => {
    setCheckedResults(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  // Export checked results to CSV (không header, không dấu nháy)
  const handleExportCSV = () => {
    const selected = results.filter(r => checkedResults.includes(r.key));
    if (selected.length === 0) return;
    const rows = selected.map(r => {
      const [keyframe, ...rest] = r.title.split(' ');
      return [keyframe, rest.join(' ')];
    });
    const csv = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{
      marginLeft: showVideoSelector ? '270px' : '0px',
      padding: '20px',
      height: '100vh',
      display: 'flex',
      flexDirection: 'row',
      gap: '24px'
    }}>
      {/* Chat Box */}
      <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <h2>💬 {currentVideo ? `Chat with ${currentVideo.original_name}` : 'Chat'}</h2>
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
      {/* MessageList panel: chỉ hiển thị tin nhắn người dùng nhập */}
      <div style={{ flex: 1, minWidth: 0, background: '#fff', borderRadius: '8px', padding: 24, boxShadow: '0 2px 8px #f0f1f2', height: '100%', display: 'flex', flexDirection: 'column', marginBottom: 0, gap: 16 }}>
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginTop: 0, marginBottom: 16 }}>📝 Lịch sử tin nhắn đã nhập</h3>
          <MessageList
            messages={messages.filter(m => m.type === 'user')}
            sessionId={currentVideo?.file_id || 'default'}
          />
        </div>
        {showVideo && (
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ShowVideo
              videoUrl={showVideo.videoUrl}
              time={showVideo.time}
              title={showVideo.title}
              onClose={() => setShowVideo(null)}
            />
          </div>
        )}
      </div>
      {/* Result panel: luôn hiển thị bên phải */}
      <div style={{
        flex: 1,
        minWidth: 0,
        background: '#f8fbff',
        borderRadius: 12,
        padding: 20,
        boxShadow: '0 2px 8px #e0e4ea',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        marginBottom: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 22 }}>🔍</span>
          <span style={{ fontWeight: 600, fontSize: 20 }}>Search Results</span>
          <button
            onClick={handleExportCSV}
            style={{
              marginLeft: 'auto',
              padding: '6px 16px',
              background: '#007bff',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              fontWeight: 500,
              fontSize: 15,
              cursor: 'pointer',
              boxShadow: '0 1px 4px #e0e4ea',
              transition: 'background 0.2s',
            }}
            disabled={checkedResults.length === 0}
            title={checkedResults.length === 0 ? 'Chọn kết quả để xuất CSV' : 'Export các kết quả đã chọn'}
          >
            Export CSV
          </button>
        </div>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          {results.length === 0 ? (
            <div style={{ color: '#888', fontStyle: 'italic', marginTop: 12 }}>No results found</div>
          ) : (
            results.map((item) => (
              <div
                key={item.key}
                style={{
                  background: '#fff',
                  borderRadius: 8,
                  boxShadow: '0 1px 4px #e0e4ea',
                  padding: 14,
                  marginBottom: 12,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                  border: '1px solid #e3eaf2',
                  cursor: 'pointer',
                  transition: 'box-shadow 0.2s',
                }}
                onClick={e => {
                  // Đừng trigger khi click vào checkbox
                  if (e.target.type !== 'checkbox') setShowVideo({ videoUrl: item.videoUrl, time: item.time, title: item.title });
                }}
                title="Xem video tại thời điểm này"
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 500, fontSize: 16 }}>
                  <input
                    type="checkbox"
                    checked={checkedResults.includes(item.key)}
                    onChange={() => handleCheckResult(item.key)}
                    style={{ marginRight: 8, accentColor: '#007bff', width: 18, height: 18 }}
                    onClick={e => e.stopPropagation()}
                  />
                  <img
                    src={item.frameImg}
                    alt="frame"
                    style={{ width: 36, height: 36, objectFit: 'cover', borderRadius: 6, background: '#eee', marginRight: 6 }}
                  />
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.title}</span>
                </div>
                <div style={{ fontSize: 14, color: '#333' }}>{item.time} - {item.confidence} confidence</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default Chat;