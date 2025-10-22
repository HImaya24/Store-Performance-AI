import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Fab,
  Slide,
  Card,
  CardContent,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  Send as SendIcon,
  Close as CloseIcon,
  SmartToy as BotIcon,
  Person as UserIcon,
  AutoAwesome as AIIcon
} from '@mui/icons-material';
import { chatWithAI } from '../../services/api';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi there! I'm your AI retail assistant! 🤖 I can help you analyze sales data, compare store performance, identify trends, and answer complex questions about your business. What would you like to know?",
      isBot: true,
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      text: inputText,
      isBot: false,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      // Prepare conversation history
      const history = messages
        .filter(msg => msg.id !== 1) // exclude initial message
        .map(msg => ({
          text: msg.text,
          isUser: !msg.isBot
        }));

      // Get AI response
      const aiResponse = await chatWithAI(inputText, history);
      
      const botMessage = {
        id: Date.now() + 1,
        text: aiResponse.response,
        isBot: true,
        timestamp: new Date(),
        isAI: true
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
        isBot: true,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleCloseChat = () => {
    setIsOpen(false);
  };

  const quickQuestions = [
    "Which store is performing best and why?",
    "What are our top selling products this month?",
    "Show me sales trends for the last quarter",
    "Compare New York and Los Angeles store performance",
    "What insights can you give about customer behavior?"
  ];

  return (
    <>
      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="ai-chat"
        onClick={() => setIsOpen(true)}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #047857 0%, #059669 100%)',
          }
        }}
      >
        <AIIcon />
      </Fab>

      {/* Chat Window - Simple positioning */}
      <Slide direction="left" in={isOpen} mountOnEnter unmountOnExit>
        <Paper
          sx={{
            position: 'fixed',
            bottom: 24, // Fixed to bottom
            right: 24,
            width: 500,
            height: 600, // Fixed height
            display: 'flex',
            flexDirection: 'column',
            background: 'rgba(30, 41, 59, 0.98)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 2,
            zIndex: 9999,
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0,0,0,0.5)'
          }}
        >
          {/* Header with Close Button */}
          <Box
            sx={{
              p: 2,
              background: 'rgba(30, 41, 59, 0.95)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: '1px solid rgba(255,255,255,0.1)',
              minHeight: '60px'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AIIcon sx={{ color: '#10b981' }} />
              <Box>
                <Typography variant="h6" fontWeight="bold" color="#10b981">
                  AI Retail Assistant
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.7, color: '#d1fae5' }}>
                  Powered by Groq LLM • Real-time Data Analysis
                </Typography>
              </Box>
            </Box>
            <IconButton
              size="small"
              sx={{ 
                color: '#9ca3af',
                background: 'rgba(255,255,255,0.1)',
                '&:hover': {
                  background: 'rgba(239, 68, 68, 0.2)',
                  color: '#ef4444'
                }
              }}
              onClick={handleCloseChat}
            >
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Messages Area */}
          <Box
            sx={{
              flex: 1,
              p: 2,
              overflow: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%)'
            }}
          >
            {messages.map((message) => (
              <Box
                key={message.id}
                sx={{
                  display: 'flex',
                  justifyContent: message.isBot ? 'flex-start' : 'flex-end',
                  gap: 1
                }}
              >
                {message.isBot && (
                  <Avatar sx={{ 
                    width: 32, 
                    height: 32, 
                    bgcolor: message.isAI ? 'rgba(16, 185, 129, 0.2)' : 'rgba(102, 126, 234, 0.2)',
                    border: message.isAI ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(102, 126, 234, 0.3)',
                    color: message.isAI ? '#10b981' : '#667eea'
                  }}>
                    {message.isAI ? <AIIcon fontSize="small" /> : <BotIcon fontSize="small" />}
                  </Avatar>
                )}
                <Card
                  sx={{
                    maxWidth: '90%',
                    background: message.isBot 
                      ? (message.isAI 
                          ? 'rgba(16, 185, 129, 0.05)' 
                          : 'rgba(102, 126, 234, 0.05)')
                      : 'rgba(59, 130, 246, 0.05)',
                    border: message.isBot
                      ? (message.isAI 
                          ? '1px solid rgba(16, 185, 129, 0.2)' 
                          : '1px solid rgba(102, 126, 234, 0.2)')
                      : '1px solid rgba(59, 130, 246, 0.2)',
                    backdropFilter: 'blur(10px)'
                  }}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        whiteSpace: 'pre-wrap',
                        lineHeight: 1.5,
                        color: '#ffffff'
                      }}
                    >
                      {message.text}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ 
                        mt: 1, 
                        display: 'block',
                        color: message.isAI ? 'rgba(16, 185, 129, 0.7)' : 'rgba(156, 163, 175, 0.7)'
                      }}
                    >
                      {message.timestamp.toLocaleTimeString([], { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                      {message.isAI && ' • 🤖 AI Powered'}
                    </Typography>
                  </CardContent>
                </Card>
                {!message.isBot && (
                  <Avatar sx={{ 
                    width: 32, 
                    height: 32, 
                    bgcolor: 'rgba(59, 130, 246, 0.2)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    color: '#3b82f6'
                  }}>
                    <UserIcon fontSize="small" />
                  </Avatar>
                )}
              </Box>
            ))}
            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'flex-start', gap: 1 }}>
                <Avatar sx={{ 
                  width: 32, 
                  height: 32, 
                  bgcolor: 'rgba(16, 185, 129, 0.2)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  color: '#10b981'
                }}>
                  <AIIcon fontSize="small" />
                </Avatar>
                <Card sx={{ 
                  background: 'rgba(16, 185, 129, 0.05)',
                  border: '1px solid rgba(16, 185, 129, 0.2)',
                  backdropFilter: 'blur(10px)'
                }}>
                  <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} sx={{ color: '#10b981' }} />
                      <Typography variant="body2" sx={{ color: '#ffffff' }}>
                        AI is thinking...
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Box>

          {/* Quick Questions - Only show if few messages */}
          {messages.length <= 2 && (
            <Box sx={{ 
              p: 2, 
              borderBottom: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(15, 23, 42, 0.5)'
            }}>
              <Typography variant="caption" sx={{ mb: 1, display: 'block', color: '#9ca3af' }}>
                Try asking:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {quickQuestions.map((question, index) => (
                  <Chip
                    key={index}
                    label={question}
                    size="small"
                    onClick={() => setInputText(question)}
                    variant="outlined"
                    sx={{ 
                      fontSize: '0.7rem',
                      maxWidth: '100%',
                      height: 'auto',
                      py: 0.5,
                      color: '#10b981',
                      borderColor: 'rgba(16, 185, 129, 0.3)',
                      background: 'rgba(16, 185, 129, 0.05)',
                      '&:hover': {
                        background: 'rgba(16, 185, 129, 0.1)',
                        borderColor: '#10b981'
                      },
                      '& .MuiChip-label': { 
                        whiteSpace: 'normal',
                        textAlign: 'left',
                        color: '#10b981'
                      }
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Input Area - Always visible at bottom */}
          <Box sx={{ 
            p: 2, 
            borderTop: '1px solid rgba(255,255,255,0.1)',
            background: 'rgba(15, 23, 42, 0.8)'
          }}>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
              <TextField
                fullWidth
                size="small"
                placeholder="Ask me anything about your retail data..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
                multiline
                maxRows={3}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: 2,
                    '& fieldset': {
                      borderColor: 'rgba(255,255,255,0.2)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(16, 185, 129, 0.3)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#10b981',
                    },
                  },
                  '& .MuiInputBase-input::placeholder': {
                    color: 'rgba(255,255,255,0.5)',
                    opacity: 1,
                  },
                }}
              />
              <IconButton
                onClick={handleSendMessage}
                disabled={!inputText.trim() || loading}
                sx={{
                  background: 'rgba(16, 185, 129, 0.2)',
                  color: '#10b981',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  width: 40,
                  height: 40,
                  '&:hover': {
                    background: 'rgba(16, 185, 129, 0.3)',
                    transform: 'translateY(-1px)',
                  },
                  '&:disabled': {
                    background: 'rgba(255,255,255,0.1)',
                    color: 'rgba(255,255,255,0.3)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    transform: 'none',
                  }
                }}
              >
                <SendIcon fontSize="small" />
              </IconButton>
            </Box>
          </Box>
        </Paper>
      </Slide>
    </>
  );
};

export default ChatBot;