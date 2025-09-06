// src/components/Chatbot/ChatbotWidget.jsx - Professional chatbot with proper styling
import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send, Bot, User, Minimize2, Maximize2, RotateCcw } from 'lucide-react';
import { chatbotService } from '../../services/chatbot.js';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hi! I\'m your CI/CD Intelligence Assistant. I can help you analyze pipeline issues, troubleshoot errors, and provide insights about your deployments. What would you like to know?',
      timestamp: new Date(),
      sources: 0
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isMinimized) {
      inputRef.current?.focus();
    }
  }, [isOpen, isMinimized]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setIsTyping(true);

    try {
      const response = await chatbotService.sendMessage(inputMessage);
      
      // Simulate typing delay for better UX
      setTimeout(() => {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: response.response || 'I apologize, but I encountered an issue processing your request. Please try again.',
          timestamp: new Date(),
          sources: response.relevant_knowledge || 0
        };

        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
      }, 1000);
      
    } catch (error) {
      console.error('Chatbot error:', error);
      
      setTimeout(() => {
        const errorMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: 'I\'m having trouble connecting to my knowledge base right now. Please check that all services are running and try again.',
          timestamp: new Date(),
          sources: 0,
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsTyping(false);
      }, 1000);
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

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: 'Chat cleared! How can I help you with your CI/CD pipeline today?',
        timestamp: new Date(),
        sources: 0
      }
    ]);
  };

  const quickQuestions = [
    "What Docker errors do we have?",
    "Show me Jenkins failures",
    "How to fix deployment issues?",
    "What are common build problems?",
    "Help with Kubernetes errors"
  ];

  const handleQuickQuestion = (question) => {
    setInputMessage(question);
    setTimeout(() => sendMessage(), 100);
  };

  return (
    <>
      {/* Chatbot Toggle Button */}
      <button
        className={`chatbot-toggle-professional ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="Open CI/CD Assistant"
      >
        {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
        {!isOpen && (
          <div className="chatbot-notification-badge">
            <span>AI</span>
          </div>
        )}
      </button>

      {/* Professional Chatbot Interface */}
      {isOpen && (
        <div className={`chatbot-container-professional ${isMinimized ? 'minimized' : ''}`}>
          {/* Enhanced Header */}
          <div className="chatbot-header-professional">
            <div className="chatbot-title-section">
              <div className="chatbot-avatar">
                <Bot size={20} />
              </div>
              <div className="chatbot-info">
                <h3 className="chatbot-title">CI/CD Assistant</h3>
                <div className="chatbot-status">
                  <div className="status-dot online"></div>
                  <span>Online • AI-Powered</span>
                </div>
              </div>
            </div>
            
            <div className="chatbot-controls">
              <button
                className="chatbot-control-btn"
                onClick={clearChat}
                title="Clear chat"
              >
                <RotateCcw size={16} />
              </button>
              
              <button
                className="chatbot-control-btn"
                onClick={() => setIsMinimized(!isMinimized)}
                title={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
              </button>
              
              <button
                className="chatbot-control-btn close"
                onClick={() => setIsOpen(false)}
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Messages Area */}
              <div className="chatbot-messages-professional">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message-professional ${message.type} ${message.isError ? 'error' : ''}`}
                  >
                    <div className="message-avatar">
                      {message.type === 'bot' ? (
                        <Bot size={16} />
                      ) : (
                        <User size={16} />
                      )}
                    </div>
                    
                    <div className="message-content-professional">
                      <div className="message-text">
                        {message.content}
                      </div>
                      
                      {message.sources > 0 && (
                        <div className="message-sources-professional">
                          📚 Found {message.sources} relevant knowledge sources
                        </div>
                      )}
                      
                      <div className="message-timestamp-professional">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Typing Indicator */}
                {isTyping && (
                  <div className="message-professional bot typing">
                    <div className="message-avatar">
                      <Bot size={16} />
                    </div>
                    <div className="message-content-professional">
                      <div className="typing-indicator-professional">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Questions */}
              {messages.length <= 1 && (
                <div className="quick-questions-professional">
                  <div className="quick-questions-title">Quick Questions:</div>
                  <div className="quick-questions-grid">
                    {quickQuestions.map((question, index) => (
                      <button
                        key={index}
                        className="quick-question-btn"
                        onClick={() => handleQuickQuestion(question)}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Enhanced Input Area */}
              <div className="chatbot-input-professional">
                <div className="input-container-professional">
                  <textarea
                    ref={inputRef}
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about CI/CD issues, errors, deployments..."
                    rows="1"
                    disabled={isLoading}
                    className="message-input-professional"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isLoading}
                    className="send-button-professional"
                  >
                    <Send size={16} />
                  </button>
                </div>
                
                <div className="input-footer-professional">
                  <span className="input-hint">
                    Press Enter to send • Shift+Enter for new line
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};

export default ChatbotWidget;
