// services/api.js
import axios from 'axios';
import { authService } from './auth';

const AGENT_ENDPOINTS = {
  collector: "http://localhost:8100",
  coordinator: "http://localhost:8110", 
  analyzer: "http://localhost:8101",
  kpi: "http://localhost:8102",
  report: "http://localhost:8103"
};

export const api = axios.create({
  timeout: 30000,
});

// Add request interceptor to automatically include auth token
api.interceptors.request.use(
  (config) => {
    const token = authService.getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      authService.logout();
      // You can redirect to login page here if needed
      console.error('Authentication failed - please login again');
    }
    return Promise.reject(error);
  }
);

// Agent status check
export const checkAgentStatus = async (agentName) => {
  try {
    const response = await api.get(`${AGENT_ENDPOINTS[agentName]}/health`);
    return { status: 'online', lastChecked: new Date().toISOString() };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required');
    }
    return { status: 'offline', lastChecked: new Date().toISOString() };
  }
};

// Check all agents
export const checkAllAgents = async () => {
  const agents = {};
  for (const agentName of Object.keys(AGENT_ENDPOINTS)) {
    agents[agentName] = await checkAgentStatus(agentName);
  }
  return agents;
};

// Load data from collector
export const loadDataFromCollector = async () => {
  try {
    const response = await api.get(`${AGENT_ENDPOINTS.collector}/events`);
    return { data: response.data, fromCollector: true };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required. Please login again.');
    }
    
    // Generate sample data like your Streamlit fallback
    const dates = Array.from({ length: 100 }, (_, i) => 
      new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString()
    );
    const products = ['coffee maker', 'blender', 'toaster', 'microwave', 'air fryer', 'rice cooker'];
    const stores = ['Los Angeles', 'New York', 'Chicago', 'Miami', 'Seattle'];
    
    const data = dates.map((date, i) => ({
      event_id: `event_${i}`,
      store_id: stores[Math.floor(Math.random() * stores.length)],
      ts: date,
      event_type: ['sale', 'inventory', 'visit', 'return', 'restock'][Math.floor(Math.random() * 5)],
      payload: {
        amount: Math.random() * 500 + 10,
        items: [products[Math.floor(Math.random() * products.length)]],
        customer_category: ['VIP', 'Regular', 'New'][Math.floor(Math.random() * 3)],
        payment_method: ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'][Math.floor(Math.random() * 4)],
        season: ['winter', 'spring', 'summer', 'fall'][Math.floor(Math.random() * 4)]
      }
    }));
    
    return { data, fromCollector: false };
  }
};

// Semantic Search
export const semanticSearch = async (query) => {
  try {
    const response = await api.post(`${AGENT_ENDPOINTS.analyzer}/semantic-search`, null, {
      params: { query }
    });
    return response.data;
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required. Please login again.');
    }
    return { error: error.message, results: [] };
  }
};

// Trigger coordinator processing
export const triggerDataProcessing = async (processType) => {
  try {
    // First get data from collector
    const collectorResponse = await api.get(`${AGENT_ENDPOINTS.collector}/events`);
    let events = collectorResponse.data;

    // Limit events to coordinator requirement
    if (events.length > 20) {
      events = events.slice(0, 20);
    }

    // Send to coordinator with proper authentication
    const response = await api.post(`${AGENT_ENDPOINTS.coordinator}/orchestrate`, 
      { events },
      {
        headers: {
          'X-API-KEY': 'demo-key'
        }
      }
    );
    
    return { success: true, data: response.data };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      return { success: false, error: 'Authentication failed. Please login again.' };
    }
    return { success: false, error: error.message };
  }
};

// Get analysis from analyzer
export const getAnalysis = async (analysisType) => {
  try {
    // First get data from collector
    const collectorResponse = await api.get(`${AGENT_ENDPOINTS.collector}/events`);
    let events = collectorResponse.data;

    // Limit to reasonable number for testing
    if (events.length > 10) {
      events = events.slice(0, 10);
    }

    // Send events array directly (not wrapped in object)
    const response = await api.post(`${AGENT_ENDPOINTS.analyzer}/analyze`, events);
    
    return { success: true, data: response.data };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required. Please login again.');
    }
    if (error.response?.status === 422) {
      return { 
        success: false, 
        error: `Data format error (422): ${error.response.data?.detail || 'Check event data structure'}` 
      };
    }
    return { success: false, error: error.message };
  }
};

// Get KPIs
export const getKPIs = async () => {
  try {
    const response = await api.get(`${AGENT_ENDPOINTS.kpi}/kpis`);
    return { success: true, data: response.data };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required. Please login again.');
    }
    return { success: false, error: error.message };
  }
};

// Generate report
export const generateReport = async (storeId) => {
  try {
    const response = await api.get(`${AGENT_ENDPOINTS.report}/report/${storeId}`);
    return { success: true, data: response.data };
  } catch (error) {
    if (error.response?.status === 401) {
      authService.logout();
      throw new Error('Authentication required. Please login again.');
    }
    return { success: false, error: error.message };
  }
};

// Login function
export const login = async (username, password) => {
  try {
    const response = await api.post(`${AGENT_ENDPOINTS.collector}/login`, {
      username,
      password
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.detail || 'Login failed' 
    };
  }
};

// Logout function
export const logout = () => {
  authService.logout();
};

// Export auth service for use in components
export { authService };