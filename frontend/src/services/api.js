// api.js - API client để interact với AI agents
import axios from 'axios';

// Configure base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance với default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor để add auth headers nếu cần
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token nếu có
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor để handle errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Chat API - interact với ConversationOrchestrator
 */
export const chatAPI = {
  /**
   * Send message và nhận response từ agents
   * @param {string} message - User message
   * @param {string} sessionId - Session ID
   * @param {string} userId - User ID
   * @returns {Promise} Response từ ConversationOrchestrator
   */
  async sendMessage(message, sessionId, userId) {
    try {
      const response = await apiClient.post('/chat/message', {
        message,
        session_id: sessionId,
        user_id: userId
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  },

  /**
   * Get chat history cho session
   * @param {string} sessionId - Session ID
   * @returns {Promise} Chat history
   */
  async getChatHistory(sessionId) {
    try {
      const response = await apiClient.get(`/chat/history/${sessionId}`);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  }
};

/**
 * Video API - interact với PreprocessingOrchestrator
 */
export const videoAPI = {
  /**
   * Upload video file
   * @param {File} file - Video file
   * @param {Function} onProgress - Upload progress callback
   * @returns {Promise} Upload result
   */
  async uploadVideo(file, onProgress) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post('/video/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress && onProgress(progress);
        },
      });

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  },

  /**
   * Process single video through preprocessing pipeline
   * @param {string} videoPath - Path to video file
   * @param {Object} config - Processing configuration
   * @returns {Promise} Processing result
   */
  async processVideo(videoPath, config = {}) {
    try {
      const response = await apiClient.post('/video/process', {
        video_path: videoPath,
        config
      });

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  },

  /**
   * Process batch videos
   * @param {Array} videoPaths - Array of video paths
   * @param {number} parallelWorkers - Number of parallel workers
   * @param {Object} config - Processing configuration
   * @returns {Promise} Batch processing result
   */
  async processBatch(videoPaths, parallelWorkers = 4, config = {}) {
    try {
      const response = await apiClient.post('/video/batch-process', {
        video_paths: videoPaths,
        parallel_workers: parallelWorkers,
        config
      });

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  },

  /**
   * Get processing status
   * @param {string} jobId - Job ID
   * @returns {Promise} Job status
   */
  async getProcessingStatus(jobId) {
    try {
      const response = await apiClient.get(`/video/status/${jobId}`);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  }
};

/**
 * System API - health checks và system info
 */
export const systemAPI = {
  /**
   * Health check
   * @returns {Promise} System health status
   */
  async healthCheck() {
    try {
      const response = await apiClient.get('/health');
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  },

  /**
   * Get system info
   * @returns {Promise} System information
   */
  async getSystemInfo() {
    try {
      const response = await apiClient.get('/');
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      };
    }
  }
};

export default apiClient;
