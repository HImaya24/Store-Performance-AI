// components/dashboard/Dashboard.js
import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Container,
  Grid,
  Card,
  CardContent,
  Alert,
  IconButton,
  Menu,
  MenuItem,
  Button,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  CloudDownload as CollectorIcon,
  Sync as CoordinatorIcon,
  Analytics as AnalyzerIcon,
  Search as ProductSearchIcon,
  FindInPage as IRSearchIcon,
  TrendingUp as KpiIcon,
  Assessment as ReportIcon,
  Logout as LogoutIcon,
  AccountCircle as AccountIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Sync as SyncIcon
} from '@mui/icons-material';

import AgentStatus from './AgentStatus';
import CollectorTab from './CollectorTab';
import CoordinatorTab from './CoordinatorTab';
import AnalyzerTab from './AnalyzerTab';
import ProductSearchTab from './ProductSearchTab';
import IRSearchTab from './IRSearchTab';
import KpiTab from './KpiTab';
import ReportTab from './ReportTab';
import { 
  checkAllAgents, 
  loadDataFromCollector, 
  authService,
  triggerDataProcessing,
  getAnalysis,
  getKPIs,
  generateReport,
  semanticSearch
} from '../../services/api';
import PricingTab from './PricingTab';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import ChatBot from './ChatBot';

const TabPanel = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`dashboard-tabpanel-${index}`}
    aria-labelledby={`dashboard-tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
  </div>
);

const Dashboard = ({ onLogout }) => {
  const [tabValue, setTabValue] = useState(0);
  const [agents, setAgents] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [processingResult, setProcessingResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [kpiData, setKpiData] = useState(null);
  const [searchResults, setSearchResults] = useState(null);

  const tabs = [
    { label: 'Collector', icon: <CollectorIcon />, component: <CollectorTab data={data} onDataUpdate={setData} /> },
    { label: 'Coordinator', icon: <CoordinatorIcon />, component: <CoordinatorTab /> },
    { label: 'Analyzer', icon: <AnalyzerIcon />, component: <AnalyzerTab /> },
    { label: 'Product Search', icon: <ProductSearchIcon />, component: <ProductSearchTab data={data} /> },
    { label: 'IR Search', icon: <IRSearchIcon />, component: <IRSearchTab /> },
    { label: 'KPI', icon: <KpiIcon />, component: <KpiTab /> },
    { label: 'Report', icon: <ReportIcon />, component: <ReportTab data={data} /> },
    { label: 'Pricing', icon: <AttachMoneyIcon />, component: <PricingTab /> },
  ];

  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      try {
        const [agentsData, collectorData] = await Promise.all([
          checkAllAgents(),
          loadDataFromCollector()
        ]);
        setAgents(agentsData);
        setData(collectorData);
        
        // Load KPIs on startup
        const kpiResult = await getKPIs();
        if (kpiResult.success) {
          setKpiData(kpiResult.data);
        }
      } catch (error) {
        console.error('Failed to initialize data:', error);
      } finally {
        setLoading(false);
      }
    };

    initializeData();

    // Refresh agent status every 30 seconds
    const interval = setInterval(async () => {
      const agentsData = await checkAllAgents();
      setAgents(agentsData);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleRefreshAgents = async () => {
    const agentsData = await checkAllAgents();
    setAgents(agentsData);
  };

  const handleUserMenuOpen = (event) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleLogout = () => {
    handleUserMenuClose();
    onLogout();
  };

  const handleProcessData = async () => {
    if (!authService.isAuthenticated()) {
      alert('Please login to process data');
      return;
    }

    setProcessing(true);
    setProcessingResult(null);
    
    try {
      const result = await triggerDataProcessing('full');
      setProcessingResult(result);
    } catch (error) {
      setProcessingResult({ 
        success: false, 
        error: error.message || 'Processing failed' 
      });
    } finally {
      setProcessing(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!authService.isAuthenticated()) {
      alert('Please login to run analysis');
      return;
    }

    setProcessing(true);
    setAnalysisResult(null);
    
    try {
      const result = await getAnalysis('insights');
      setAnalysisResult(result);
    } catch (error) {
      setAnalysisResult({ 
        success: false, 
        error: error.message || 'Analysis failed' 
      });
    } finally {
      setProcessing(false);
    }
  };

  const handleRefreshKPIs = async () => {
    try {
      const result = await getKPIs();
      if (result.success) {
        setKpiData(result.data);
      }
    } catch (error) {
      console.error('Failed to refresh KPIs:', error);
    }
  };

  const handleSearch = async (query) => {
    if (!authService.isAuthenticated()) {
      alert('Please login to search');
      return;
    }

    try {
      const result = await semanticSearch(query);
      setSearchResults(result);
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  const getOnlineAgentCount = () => {
    return Object.values(agents).filter(agent => agent.status === 'online').length;
  };

  const getTotalAgentCount = () => {
    return Object.keys(agents).length;
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4, textAlign: 'center' }}>
        <CircularProgress size={60} sx={{ color: 'primary.main', mb: 2 }} />
        <Typography variant="h4" color="white">Loading Store AI Dashboard...</Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
          Initializing security and loading data...
        </Typography>
      </Container>
    );
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.90)), url('https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
      }}
    >
      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Header with User Menu */}
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
              <Typography variant="h3" fontWeight="bold" color="primary">
                🏪 Store Performance AI Dashboard
              </Typography>
            </Box>
            <Typography variant="h6" color="text.secondary">
              Monitor your retail AI agents and analyze transaction data
            </Typography>
          </Box>
          
          {/* User Menu */}
          <Box>
            <IconButton
              onClick={handleUserMenuOpen}
              sx={{
                color: 'white',
                background: 'rgba(255,255,255,0.1)',
                '&:hover': {
                  background: 'rgba(255,255,255,0.2)',
                }
              }}
            >
              <AccountIcon />
            </IconButton>
            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={handleUserMenuClose}
              PaperProps={{
                sx: {
                  background: 'rgba(30, 41, 59, 0.9)',
                  backdropFilter: 'blur(10px)',
                }
              }}
            >
              <MenuItem onClick={handleLogout}>
                <LogoutIcon sx={{ mr: 1 }} />
                Logout
              </MenuItem>
            </Menu>
          </Box>
        </Box>

        
        {/* Processing Results */}
        {processingResult && (
          <Alert 
            severity={processingResult.success ? "success" : "error"}
            sx={{ mb: 3 }}
            onClose={() => setProcessingResult(null)}
          >
            {processingResult.success ? 
              `Processing completed! Batch ID: ${processingResult.data?.batch_id}` : 
              `Processing failed: ${processingResult.error}`
            }
          </Alert>
        )}

        {analysisResult && (
          <Alert 
            severity={analysisResult.success ? "success" : "error"}
            sx={{ mb: 3 }}
            onClose={() => setAnalysisResult(null)}
          >
            {analysisResult.success ? 
              `Analysis completed! ${analysisResult.data?.insights} insights generated` : 
              `Analysis failed: ${analysisResult.error}`
            }
          </Alert>
        )}

        {/* System Status Overview */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ 
              background: 'rgba(30, 41, 59, 0.8)', 
              backdropFilter: 'blur(10px)',
              border: getOnlineAgentCount() === getTotalAgentCount() ? 
                '1px solid #10b981' : '1px solid #f59e0b'
            }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                  {getOnlineAgentCount() === getTotalAgentCount() ? 
                    <CheckCircleIcon sx={{ color: '#10b981', mr: 1 }} /> :
                    <WarningIcon sx={{ color: '#f59e0b', mr: 1 }} />
                  }
                  <Typography variant="h4" fontWeight="bold" 
                    color={getOnlineAgentCount() === getTotalAgentCount() ? "success.main" : "warning.main"}>
                    {getOnlineAgentCount()}/{getTotalAgentCount()}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Agents Online
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="primary">
                  {data?.data ? data.data.length : 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Events Loaded
                </Typography>
                <Chip 
                  label={data?.fromCollector ? "Live Data" : "Sample Data"} 
                  color={data?.fromCollector ? "success" : "warning"}
                  size="small"
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="secondary">
                {data?.data ? [...new Set(data.data.map(item => item.store_id))].length : 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Stores Tracked
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <CheckCircleIcon sx={{ color: '#10b981', mr: 1 }} />
                  <Typography variant="h4" fontWeight="bold" color="success.main">
                    Secure
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Authentication Active
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Agent Status */}
        <Box sx={{ mb: 4 }}>
          <AgentStatus agents={agents} onRefresh={handleRefreshAgents} />
        </Box>

        {/* Main Tabs */}
        <Paper sx={{ 
          background: 'rgba(30, 41, 59, 0.8)', 
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.1)'
        }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            aria-label="dashboard tabs"
            sx={{
              borderBottom: 1,
              borderColor: 'divider',
              '& .MuiTab-root': {
                minHeight: 64,
                fontWeight: 600,
                color: 'rgba(255,255,255,0.7)',
                '&.Mui-selected': {
                  color: 'primary.main',
                }
              }
            }}
          >
            {tabs.map((tab, index) => (
              <Tab
                key={tab.label}
                icon={tab.icon}
                label={tab.label}
                iconPosition="start"
                id={`dashboard-tab-${index}`}
                aria-controls={`dashboard-tabpanel-${index}`}
              />
            ))}
          </Tabs>

          {tabs.map((tab, index) => (
            <TabPanel key={tab.label} value={tabValue} index={index}>
              {React.cloneElement(tab.component, { 
                data,
                onProcessData: handleProcessData,
                onRunAnalysis: handleRunAnalysis,
                onRefreshKPIs: handleRefreshKPIs,
                onSearch: handleSearch,
                processing,
                processingResult,
                analysisResult,
                kpiData,
                searchResults
              })}
            </TabPanel>
          ))}
        </Paper>

        {/* Security Footer */}
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            🔒 Secured Session • Role: User • {new Date().toLocaleDateString()}
          </Typography>
        </Box>
      </Container>

      <ChatBot />
    </Box>
  );
};

export default Dashboard;