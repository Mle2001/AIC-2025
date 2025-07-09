import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatAPI } from '../services/api';
import { useWebSocket } from './useWebSocket';
import { useAuth } from './useAuth';

/**
 * useChat - Custom hook quản lý toàn bộ logic chat, WebSocket, message history, optimistic update, error recovery
 * @param {string} sessionId - ID của phiên chat
 * @returns {object} { messages, sendMessage, isLoading, error, connectionStatus, isTyping, loadHistory }
 */
export function useChat(sessionId) {
  const { user } = useAuth();
  const userId = user?.id || 'user1';
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const queryClient = useQueryClient();

  // WebSocket hook (abstracted)
  const { ws, send: wsSend, status: wsStatus } = useWebSocket(`/chat/ws/${sessionId}`, {
    onMessage: (data) => {
      handleIncomingMessage(data);
    },
    onStatus: (status) => {
      setConnectionStatus(status);
      if (status === 'reconnecting') setIsTyping(false);
    },
    onError: (err) => setError(err?.message || 'WebSocket error'),
  });

  // Sync wsRef
  useEffect(() => {
    wsRef.current = ws;
  }, [ws]);

  // Handle incoming message
  const handleIncomingMessage = useCallback((data) => {
    const { type, content } = data;
    if (type === 'user_message') {
      setMessages((prev) => [...prev, { ...data, isUser: true }]);
      setIsTyping(true);
    } else if (type === 'bot_response') {
      setMessages((prev) => [...prev, { ...data, isUser: false }]);
      setIsTyping(false);
    } else if (type === 'error') {
      setError(content || 'Chat error');
      setIsTyping(false);
    }
  }, []);

  // REST API: load history
  const { isLoading: isHistoryLoading, refetch: loadHistory } = useQuery({
    queryKey: ['chat-history', sessionId],
    queryFn: () => chatAPI.getHistory(sessionId),
    enabled: !!sessionId,
    onSuccess: (data) => {
      setMessages(data || []);
    },
    onError: (err) => setError(err?.message || 'Lỗi tải lịch sử chat'),
  });

  // Send message (WebSocket ưu tiên, fallback REST)
  const sendMessage = useCallback(async (message) => {
    setError(null);
    const msgObj = {
      message,
      user_id: userId,
      timestamp: new Date().toISOString(),
    };
    // Optimistic update
    setMessages((prev) => [
      ...prev,
      { ...msgObj, type: 'user_message', isUser: true },
    ]);
    setIsTyping(true);
    // Gửi qua WebSocket nếu có
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsSend(msgObj);
        return true;
      } catch (err) {
        setError('Gửi qua WebSocket thất bại');
      }
    }
    // Fallback REST
    try {
      await chatAPI.sendMessage(sessionId, msgObj);
      // Optionally: refetch/loadHistory();
      return true;
    } catch (err) {
      setError('Gửi message thất bại');
      setIsTyping(false);
      return false;
    }
  }, [sessionId, userId, wsSend]);

  return {
    messages,
    sendMessage,
    isLoading: isHistoryLoading,
    error,
    connectionStatus: wsStatus || connectionStatus,
    isTyping,
    loadHistory,
  };
}
