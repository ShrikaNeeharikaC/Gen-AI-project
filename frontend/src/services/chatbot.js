// src/services/chatbot.js - Chatbot API service
const CHATBOT_API_URL = 'http://localhost:5006';

class ChatbotService {
  constructor() {
    this.sessionId = this.generateSessionId();
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async sendMessage(message, sessionId = null) {
    try {
      const response = await fetch(`${CHATBOT_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          session_id: sessionId || this.sessionId,
          user_id: 'dashboard_user'
        }),
      });

      if (!response.ok) {
        throw new Error(`Chatbot API Error: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Chatbot service error:', error);
      throw error;
    }
  }

  async getChatHistory(sessionId = null) {
    try {
      const response = await fetch(
        `${CHATBOT_API_URL}/api/chat/history/${sessionId || this.sessionId}`
      );

      if (!response.ok) {
        throw new Error(`History API Error: ${response.status}`);
      }

      const data = await response.json();
      return data.history || [];
    } catch (error) {
      console.error('Chat history error:', error);
      return [];
    }
  }

  async checkHealth() {
    try {
      const response = await fetch(`${CHATBOT_API_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Chatbot health check failed:', error);
      return false;
    }
  }
}

export const chatbotService = new ChatbotService();
