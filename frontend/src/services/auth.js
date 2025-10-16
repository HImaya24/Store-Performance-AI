// services/auth.js - SIMPLE WORKING AUTH
const AUTH_TOKEN_KEY = 'auth_token';
const CURRENT_USER_KEY = 'current_user';

export const authService = {
    login: async (username, password) => {
        try {
            const response = await fetch('http://localhost:8100/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                throw new Error('Login failed - check credentials');
            }

            const data = await response.json();
            
            // Store token and user info
            localStorage.setItem(AUTH_TOKEN_KEY, data.token);
            localStorage.setItem(CURRENT_USER_KEY, JSON.stringify({
                username: data.user,
                role: data.role
            }));
            
            return data;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    },

    logout: () => {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(CURRENT_USER_KEY);
    },

    getToken: () => {
        return localStorage.getItem(AUTH_TOKEN_KEY);
    },

    getCurrentUser: () => {
        const userStr = localStorage.getItem(CURRENT_USER_KEY);
        return userStr ? JSON.parse(userStr) : null;
    },

    isAuthenticated: () => {
        return !!localStorage.getItem(AUTH_TOKEN_KEY);
    },

    // Add token to requests
    authHeaders: () => {
        const token = authService.getToken();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    }
};