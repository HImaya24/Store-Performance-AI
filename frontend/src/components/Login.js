// components/dashboard/Login.js
import React, { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  IconButton
} from '@mui/material';
import { ArrowBack as ArrowBackIcon } from '@mui/icons-material';
import { authService } from '../services/api';

const Login = ({ onLogin, onBack }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      await authService.login(username, password);
      onLogin(); // This will trigger the redirect to dashboard
    } catch (err) {
      setError('Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      }}
    >
      <Paper
        sx={{
          p: 4,
          maxWidth: 400,
          width: '100%',
          background: 'rgba(30, 41, 59, 0.9)',
          backdropFilter: 'blur(10px)',
          position: 'relative'
        }}
      >
        {/* Back Button */}
        <IconButton
          onClick={onBack}
          sx={{
            position: 'absolute',
            left: 16,
            top: 16,
            color: 'white'
          }}
        >
          <ArrowBackIcon />
        </IconButton>

        <Typography variant="h4" gutterBottom color="white" textAlign="center">
          🔐 Login
        </Typography>
        
        <Typography variant="body1" color="text.secondary" textAlign="center" sx={{ mb: 3 }}>
          Sign in to access the Store Performance Dashboard
        </Typography>
        
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            margin="normal"
            required
            disabled={loading}
            sx={{
              '& .MuiInputBase-input': {
                color: 'white'
              },
              '& .MuiInputLabel-root': {
                color: 'rgba(255, 255, 255, 0.7)'
              },
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.3)'
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.5)'
                }
              }
            }}
          />
          <TextField
            fullWidth
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            margin="normal"
            required
            disabled={loading}
            sx={{
              '& .MuiInputBase-input': {
                color: 'white'
              },
              '& .MuiInputLabel-root': {
                color: 'rgba(255, 255, 255, 0.7)'
              },
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.3)'
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.5)'
                }
              }
            }}
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ 
              mt: 3,
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              py: 1.5,
              fontSize: '1.1rem',
              '&:hover': {
                background: 'linear-gradient(135deg, #5b5cdc 0%, #7c4cf0 100%)',
              }
            }}
            disabled={loading}
          >
            {loading ? 'Signing In...' : 'Sign In'}
          </Button>
        </form>
        
        <Box sx={{ mt: 3, p: 2, background: 'rgba(255,255,255,0.1)', borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary" textAlign="center">
            Demo Credentials:
          </Typography>
          <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ mt: 1 }}>
            <strong>Admin:</strong> admin / admin123<br />
            <strong>User:</strong> user / user123
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default Login;