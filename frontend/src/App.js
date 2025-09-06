import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, IconButton, Fab } from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import Dashboard from './components/Dashboard';
import ProjectDetails from './components/ProjectDetails';
import Chatbot from './components/Chatbot';
import Header from './components/Header';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [chatOpen, setChatOpen] = useState(false);

  const toggleChat = () => setChatOpen(!chatOpen);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Header />
          <Box sx={{ flexGrow: 1, p: 3 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/project/:toolName/:projectName" element={<ProjectDetails />} />
            </Routes>
          </Box>
          
          {/* Chatbot FAB */}
          <Fab
            color="primary"
            aria-label="chat"
            onClick={toggleChat}
            sx={{
              position: 'fixed',
              bottom: 16,
              right: 16,
              zIndex: 1000,
            }}
          >
            <ChatIcon />
          </Fab>
          
          {chatOpen && <Chatbot onClose={toggleChat} />}
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
