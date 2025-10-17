// App.js
import React, { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { theme } from './themes/theme';
import Dashboard from './components/dashboard/Dashboard';
import LandingPage from './components/dashboard/LandingPage';
import Login from './components/Login'; 
import { authService } from './services/api'; 
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('landing'); 
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check if user is already logged in on app start
  useEffect(() => {
    if (authService.isAuthenticated()) {
      setIsAuthenticated(true);
      setCurrentView('dashboard');
    }
  }, []);

  const handleGetStarted = () => {
    setCurrentView('login');
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
    setCurrentView('dashboard');
  };

  const handleLogout = () => {
    authService.logout();
    setIsAuthenticated(false);
    setCurrentView('landing');
  };

  // Render different views based on current state
  const renderCurrentView = () => {
    switch (currentView) {
      case 'landing':
        return <LandingPage onGetStarted={handleGetStarted} />;
      
      case 'login':
        return <Login onLogin={handleLogin} onBack={() => setCurrentView('landing')} />;
      
      case 'dashboard':
        return <Dashboard onLogout={handleLogout} />;
      
      default:
        return <LandingPage onGetStarted={handleGetStarted} />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        }}
      >
        {renderCurrentView()}
      </Box>
    </ThemeProvider>
  );
}

export default App;