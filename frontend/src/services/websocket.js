// websocket.js - WebSocket client cho real-time chat với ConversationOrchestrator
class ChatWebSocketClient {
  constructor(sessionId, userId) {
    this.sessionId = sessionId;
    this.userId = userId;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 3000; // 3 seconds
    this.messageHandlers = new Set();
    this.statusHandlers = new Set();
  }

  /**
   * Connect tới WebSocket server
   */
  connect() {
    const wsUrl = `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000'}/chat/ws/${this.sessionId}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.handleConnectionError();
    }
  }

  /**
   * Setup WebSocket event handlers
   */
  setupEventHandlers() {
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.notifyStatusHandlers('connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      this.notifyStatusHandlers('disconnected');
      
      // Attempt reconnection nếu không phải intentional close
      if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.attemptReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.notifyStatusHandlers('error');
    };
  }

  /**
   * Handle incoming messages từ ConversationOrchestrator
   */
  handleMessage(data) {
    const { type, message, session_id, processing_time, media_references, error } = data;

    switch (type) {
      case 'response':
        // Agent response từ ConversationOrchestrator
        this.notifyMessageHandlers({
          type: 'bot_response',
          content: message,
          sessionId: session_id,
          processingTime: processing_time,
          mediaReferences: media_references || [],
          timestamp: new Date().toISOString()
        });
        break;

      case 'error':
        // Error từ agents
        this.notifyMessageHandlers({
          type: 'error',
          content: `Error: ${error || message}`,
          timestamp: new Date().toISOString()
        });
        break;

      case 'status':
        // Status updates (typing, processing, etc.)
        this.notifyStatusHandlers(message);
        break;

      default:
        console.warn('Unknown message type:', type);
    }
  }

  /**
   * Send message tới ConversationOrchestrator
   */
  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const messageData = {
        message,
        user_id: this.userId,
        timestamp: new Date().toISOString()
      };

      this.ws.send(JSON.stringify(messageData));
      
      // Notify về user message
      this.notifyMessageHandlers({
        type: 'user_message',
        content: message,
        userId: this.userId,
        timestamp: new Date().toISOString()
      });

      return true;
    } else {
      console.error('WebSocket not connected');
      return false;
    }
  }

  /**
   * Attempt reconnection
   */
  attemptReconnect() {
    this.reconnectAttempts++;
    this.notifyStatusHandlers('reconnecting');

    setTimeout(() => {
      console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect();
    }, this.reconnectInterval);
  }

  /**
   * Handle connection errors
   */
  handleConnectionError() {
    this.notifyStatusHandlers('error');
    
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.attemptReconnect();
    }
  }

  /**
   * Add message handler
   */
  onMessage(handler) {
    this.messageHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  /**
   * Add status handler
   */
  onStatus(handler) {
    this.statusHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  /**
   * Notify message handlers
   */
  notifyMessageHandlers(data) {
    this.messageHandlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error('Message handler error:', error);
      }
    });
  }

  /**
   * Notify status handlers
   */
  notifyStatusHandlers(status) {
    this.statusHandlers.forEach(handler => {
      try {
        handler(status);
      } catch (error) {
        console.error('Status handler error:', error);
      }
    });
  }

  /**
   * Disconnect WebSocket
   */
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'User disconnected');
      this.ws = null;
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'disconnecting';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }
}

export default ChatWebSocketClient;
