import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * useWebSocket - Custom hook quản lý WebSocket kết nối và sự kiện
 * @param {string} path - Đường dẫn WebSocket (ví dụ: /chat/ws/abc)
 * @param {object} options - { onMessage, onStatus, onError }
 * @returns {object} { ws, send, status }
 */
export function useWebSocket(path, options = {}) {
  const wsRef = useRef(null);
  const [status, setStatus] = useState('disconnected');
  const baseUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
  const url = path.startsWith('ws') ? path : `${baseUrl}${path}`;

  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    let ws;
    try {
      ws = new window.WebSocket(url);
      wsRef.current = ws;
      setStatus('connecting');
    } catch (err) {
      setStatus('error');
      options.onError && options.onError(err);
      return;
    }

    ws.onopen = () => {
      setStatus('connected');
      options.onStatus && options.onStatus('connected');
    };
    ws.onclose = () => {
      setStatus('disconnected');
      options.onStatus && options.onStatus('disconnected');
    };
    ws.onerror = (e) => {
      setStatus('error');
      options.onError && options.onError(e);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        options.onMessage && options.onMessage(data);
      } catch (err) {
        options.onError && options.onError(err);
      }
    };

    return () => {
      ws.close();
    };
    // eslint-disable-next-line
  }, [url]);

  return { ws: wsRef.current, send, status };
}
