// ChatWidget.jsx - Main chat widget component
import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send, User, Bot, Loader2, Settings, History } from 'lucide-react';
import './ChatWidget.css';

const ChatWidget = ({ 
  apiUrl = 'http://localhost:5000/api',
  position = 'bottom-right',
  theme = 'light',
  title = 'CI/CD Assistant',
  subtitle = 'Ask me about your pipeline issues'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Generate session ID on mount
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    checkHealth();
  }, []);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input when widget opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current.focus(), 100);
    }
  }, [isOpen]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkHealth = async () => {
    try {
      const response = await fetch(`${apiUrl}/health`);
      if (response.ok) {
        setConnectionStatus('connected');
        // Add welcome message
        setMessages([{
          id: Date.now(),
          type: 'bot',
          content: `Hello! I'm your CI/CD Assistant. I can help you troubleshoot pipeline issues, Docker problems, Kubernetes deployments, and more. What can I help you with today?`,
          timestamp: new Date(),
          intent: 'greeting'
        }]);
      } else {
        setConnectionStatus('error');
      }
    } catch (error) {
      setConnectionStatus('error');
      console.error('Health check failed:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !sessionId) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentMessage,
          session_id: sessionId,
          user_id: 'web_user'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.response,
        timestamp: new Date(),
        intent: data.intent,
        entities: data.entities,
        knowledgeSources: data.knowledge_sources,
        relevantKnowledge: data.relevant_knowledge
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error processing your request. Please check if the backend service is running and try again.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const loadChatHistory = async () => {
    if (!sessionId) return;
    
    try {
      const response = await fetch(`${apiUrl}/chat/history/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.history && data.history.length > 0) {
          const historyMessages = data.history.map(msg => ({
            id: `history_${Date.now()}_${Math.random()}`,
            type: msg.role === 'user' ? 'user' : 'bot',
            content: msg.content,
            timestamp: new Date(msg.timestamp),
            intent: msg.intent
          }));
          setMessages(prev => [...historyMessages, ...prev]);
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const clearChat = () => {
    setMessages([{
      id: Date.now(),
      type: 'bot',
      content: "Chat cleared! How can I help you with your CI/CD pipeline today?",
      timestamp: new Date(),
      intent: 'greeting'
    }]);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getPositionClass = () => {
    switch (position) {
      case 'bottom-left': return 'chat-widget-bottom-left';
      case 'top-right': return 'chat-widget-top-right';
      case 'top-left': return 'chat-widget-top-left';
      default: return 'chat-widget-bottom-right';
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#10b981';
      case 'error': return '#ef4444';
      default: return '#f59e0b';
    }
  };

  const renderMessage = (message) => {
    const isUser = message.type === 'user';
    
    return (
      <div key={message.id} className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
        <div className="message-avatar">
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>
        <div className="message-content">
          <div className={`message-bubble ${message.isError ? 'error-message' : ''}`}>
            <div className="message-text">{message.content}</div>
            {message.knowledgeSources && message.knowledgeSources.length > 0 && (
              <div className="knowledge-sources">
                <small>📚 Sources: {message.knowledgeSources.join(', ')}</small>
              </div>
            )}
            {message.intent && message.intent !== 'response' && (
              <div className="message-meta">
                <small>Intent: {message.intent}</small>
              </div>
            )}
          </div>
          <div className="message-time">
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`chat-widget ${getPositionClass()} ${theme}`}>
      {/* Chat Widget Button */}
      {!isOpen && (
        <button
          className="chat-widget-button"
          onClick={() => setIsOpen(true)}
          aria-label="Open chat"
        >
          <MessageCircle size={24} />
          <div 
            className="connection-indicator" 
            style={{ backgroundColor: getConnectionStatusColor() }}
          />
        </button>
      )}

      {/* Chat Widget Panel */}
      {isOpen && (
        <div className="chat-widget-panel">
          {/* Header */}
          <div className="chat-header">
            <div className="chat-header-info">
              <h3>{title}</h3>
              <p>{subtitle}</p>
              <div className="connection-status">
                <div 
                  className="status-dot" 
                  style={{ backgroundColor: getConnectionStatusColor() }}
                />
                <span>{connectionStatus === 'connected' ? 'Online' : connectionStatus === 'error' ? 'Offline' : 'Connecting...'}</span>
              </div>
            </div>
            <div className="chat-header-actions">
              <button
                className="header-button"
                onClick={() => setShowHistory(!showHistory)}
                title="Load History"
              >
                <History size={16} />
              </button>
              <button
                className="header-button"
                onClick={() => setShowSettings(!showSettings)}
                title="Settings"
              >
                <Settings size={16} />
              </button>
              <button
                className="header-button close-button"
                onClick={() => setIsOpen(false)}
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="settings-panel">
              <button onClick={clearChat} className="setting-button">
                Clear Chat History
              </button>
              <button onClick={checkHealth} className="setting-button">
                Test Connection
              </button>
              <button onClick={loadChatHistory} className="setting-button">
                Load Previous Chat
              </button>
            </div>
          )}

          {/* Messages */}
          <div className="chat-messages">
            {messages.map(renderMessage)}
            {isLoading && (
              <div className="message bot-message">
                <div className="message-avatar">
                  <Bot size={16} />
                </div>
                <div className="message-content">
                  <div className="message-bubble">
                    <div className="typing-indicator">
                      <Loader2 size={16} className="spinning" />
                      <span>Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="chat-input-container">
            <div className="chat-input-wrapper">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about CI/CD issues, Docker errors, Kubernetes problems..."
                className="chat-input"
                rows="1"
                disabled={isLoading || connectionStatus === 'error'}
              />
              <button
                onClick={sendMessage}
                className="send-button"
                disabled={!inputMessage.trim() || isLoading || connectionStatus === 'error'}
                aria-label="Send message"
              >
                <Send size={16} />
              </button>
            </div>
            <div className="input-hint">
              <small>💡 Try asking: "Docker build failed with exit code 125" or "Kubernetes pod won't start"</small>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;