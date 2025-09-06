import React, { useState, useEffect, useRef } from 'react';
import {
  Paper, Box, Typography, TextField, IconButton,
  List, ListItem, Avatar, Chip, CircularProgress
} from '@mui/material';
import {
  Close as CloseIcon,
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon
} from '@mui/icons-material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import axios from 'axios';

function Chatbot({ onClose }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      message: 'Hello! I\'m your CI/CD assistant. I can help you analyze build failures, suggest fixes, and answer questions about your pipelines.',
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      message: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5006/chat', {
        message: input.trim(),
        session_id: sessionId,
      });

      const botMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        message: response.data.message,
        timestamp: new Date(),
        intent: response.data.intent,
        confidence: response.data.confidence,
        knowledge_sources: response.data.knowledge_sources,
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        message: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (message) => {
    // Check if message contains code blocks
    const codeBlockRegex = /``````/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(message)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: message.substring(lastIndex, match.index)
        });
      }

      // Add code block
      parts.push({
        type: 'code',
        language: match[1] || 'bash',
        content: match[2].trim()
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < message.length) {
      parts.push({
        type: 'text',
        content: message.substring(lastIndex)
      });
    }

    return parts.length > 0 ? parts : [{ type: 'text', content: message }];
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'fixed',
        bottom: 80,
        right: 16,
        width: 400,
        height: 600,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1000,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          bgcolor: 'primary.main',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box display="flex" alignItems="center">
          <BotIcon sx={{ mr: 1 }} />
          <Typography variant="h6">CI/CD Assistant</Typography>
        </Box>
        <IconButton color="inherit" onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Messages */}
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          p: 1,
          bgcolor: 'grey.50',
        }}
      >
        <List>
          {messages.map((msg) => (
            <ListItem
              key={msg.id}
              sx={{
                display: 'flex',
                flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row',
                alignItems: 'flex-start',
                mb: 1,
              }}
            >
              <Avatar
                sx={{
                  bgcolor: msg.sender === 'user' ? 'primary.main' : 'secondary.main',
                  width: 32,
                  height: 32,
                  mx: 1,
                }}
              >
                {msg.sender === 'user' ? <PersonIcon /> : <BotIcon />}
              </Avatar>
              
              <Box
                sx={{
                  maxWidth: '70%',
                  bgcolor: msg.sender === 'user' ? 'primary.main' : 'white',
                  color: msg.sender === 'user' ? 'white' : 'text.primary',
                  borderRadius: 2,
                  p: 1.5,
                  boxShadow: 1,
                }}
              >
                {formatMessage(msg.message).map((part, index) => (
                  <Box key={index}>
                    {part.type === 'text' ? (
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {part.content}
                      </Typography>
                    ) : (
                      <Box sx={{ mt: 1 }}>
                        <SyntaxHighlighter
                          language={part.language}
                          style={tomorrow}
                          customStyle={{
                            margin: 0,
                            borderRadius: 4,
                            fontSize: '0.75rem',
                          }}
                        >
                          {part.content}
                        </SyntaxHighlighter>
                      </Box>
                    )}
                  </Box>
                ))}
                
                {msg.sender === 'bot' && msg.confidence && (
                  <Box display="flex" gap={0.5} mt={1} flexWrap="wrap">
                    <Chip
                      label={`${(msg.confidence * 100).toFixed(0)}% confident`}
                      size="small"
                      variant="outlined"
                    />
                    {msg.intent && (
                      <Chip
                        label={msg.intent}
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                )}
                
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    mt: 0.5,
                    opacity: 0.7,
                  }}
                >
                  {msg.timestamp.toLocaleTimeString()}
                </Typography>
              </Box>
            </ListItem>
          ))}
          
          {loading && (
            <ListItem>
              <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32, mx: 1 }}>
                <BotIcon />
              </Avatar>
              <Box
                sx={{
                  bgcolor: 'white',
                  borderRadius: 2,
                  p: 1.5,
                  boxShadow: 1,
                }}
              >
                <CircularProgress size={20} />
              </Box>
            </ListItem>
          )}
        </List>
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <Box
        sx={{
          p: 2,
          bgcolor: 'white',
          borderTop: 1,
          borderColor: 'divider',
        }}
      >
        <Box display="flex" gap={1}>
          <TextField
            fullWidth
            multiline
            maxRows={3}
            variant="outlined"
            size="small"
            placeholder="Ask about build failures, errors, or CI/CD best practices..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <IconButton
            color="primary"
            onClick={sendMessage}
            disabled={!input.trim() || loading}
          >
            <SendIcon />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );
}

export default Chatbot;
