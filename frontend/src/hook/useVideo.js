import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { videoAPI } from '../services/api';
import { useWebSocket } from './useWebSocket';
import { useAuth } from './useAuth';

/**
 * useVideo - Custom hook quản lý toàn bộ logic upload/process video, WebSocket, state, error recovery
 * @returns {object} { videos, uploadVideo, processVideos, isLoading, error, connectionStatus, progress, results, reloadVideos }
 */
export function useVideo() {
  const { user } = useAuth();
  const userId = user?.id || 'user1';
  const [videos, setVideos] = useState([]); // Danh sách video user đã upload
  const [progress, setProgress] = useState({}); // Tiến trình process theo videoId
  const [results, setResults] = useState({}); // Kết quả process theo videoId
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const queryClient = useQueryClient();

  // WebSocket cho process video (nếu backend hỗ trợ)
  const { ws, send: wsSend, status: wsStatus } = useWebSocket('/video/ws', {
    onMessage: (data) => handleWsMessage(data),
    onStatus: (status) => setConnectionStatus(status),
    onError: (err) => setError(err?.message || 'WebSocket error'),
  });

  useEffect(() => {
    wsRef.current = ws;
  }, [ws]);

  // REST API: load danh sách video
  const { isLoading, refetch: reloadVideos } = useQuery({
    queryKey: ['videos', userId],
    queryFn: () => videoAPI.getVideos(userId),
    enabled: !!userId,
    onSuccess: (data) => setVideos(data || []),
    onError: (err) => setError(err?.message || 'Lỗi tải danh sách video'),
  });

  // Upload video (REST)
  const uploadMutation = useMutation({
    mutationFn: (file) => videoAPI.uploadVideo(userId, file),
    onSuccess: (video) => {
      setVideos((prev) => [...prev, video]);
    },
    onError: (err) => setError(err?.message || 'Lỗi upload video'),
  });

  const uploadVideo = useCallback((file) => {
    setError(null);
    uploadMutation.mutate(file);
  }, [uploadMutation]);

  // Process videos (WebSocket ưu tiên, fallback REST)
  const processVideos = useCallback(async (videoIds) => {
    setError(null);
    // Gửi qua WebSocket nếu có
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsSend({ action: 'process', videoIds, userId });
        // Optimistic: set progress về 0
        setProgress((prev) => ({ ...prev, ...Object.fromEntries(videoIds.map(id => [id, 0])) }));
        return true;
      } catch (err) {
        setError('Gửi process qua WebSocket thất bại');
      }
    }
    // Fallback REST
    try {
      await videoAPI.processVideos(userId, videoIds);
      // Optionally: reloadVideos();
      return true;
    } catch (err) {
      setError('Process video thất bại');
      return false;
    }
  }, [userId, wsSend]);

  // Handle message từ WebSocket process
  const handleWsMessage = useCallback((data) => {
    const { type, videoId, progress: prog, result, error: errMsg } = data;
    if (type === 'progress') {
      setProgress((prev) => ({ ...prev, [videoId]: prog }));
    } else if (type === 'result') {
      setResults((prev) => ({ ...prev, [videoId]: result }));
      setProgress((prev) => ({ ...prev, [videoId]: 100 }));
    } else if (type === 'error') {
      setError(errMsg || 'Lỗi process video');
      setProgress((prev) => ({ ...prev, [videoId]: 0 }));
    }
  }, []);

  return {
    videos,
    uploadVideo,
    processVideos,
    isLoading,
    error,
    connectionStatus: wsStatus || connectionStatus,
    progress,
    results,
    reloadVideos,
  };
}
