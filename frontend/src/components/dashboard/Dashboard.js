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
} from '@mui/material';
import {
  CloudDownload as CollectorIcon,
  Sync as CoordinatorIcon,
  Analytics as AnalyzerIcon,
  Search as ProductSearchIcon,
  FindInPage as IRSearchIcon,
  TrendingUp as KpiIcon,
  Assessment as ReportIcon,
} from '@mui/icons-material';

import AgentStatus from './AgentStatus';
import CollectorTab from './CollectorTab';
import CoordinatorTab from './CoordinatorTab';
import AnalyzerTab from './AnalyzerTab';
import ProductSearchTab from './ProductSearchTab';
import IRSearchTab from './IRSearchTab';
import KpiTab from './KpiTab';
import ReportTab from './ReportTab';
import { checkAllAgents, loadDataFromCollector } from '../../services/api';

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

const Dashboard = () => {
  const [tabValue, setTabValue] = useState(0);
  const [agents, setAgents] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const tabs = [
    { label: 'Collector', icon: <CollectorIcon />, component: <CollectorTab data={data} onDataUpdate={setData} /> },
    { label: 'Coordinator', icon: <CoordinatorIcon />, component: <CoordinatorTab /> },
    { label: 'Analyzer', icon: <AnalyzerIcon />, component: <AnalyzerTab /> },
    { label: 'Product Search', icon: <ProductSearchIcon />, component: <ProductSearchTab data={data} /> },
    { label: 'IR Search', icon: <IRSearchIcon />, component: <IRSearchTab /> },
    { label: 'KPI', icon: <KpiIcon />, component: <KpiTab /> },
    { label: 'Report', icon: <ReportIcon />, component: <ReportTab data={data} /> },
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

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="h4">Loading Store AI Dashboard...</Typography>
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
        {/* Header */}
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" fontWeight="bold" gutterBottom color="primary">
            🏪 Store Performance AI Dashboard
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Monitor your retail AI agents and analyze transaction data
          </Typography>
        </Box>

        {/* Agent Status */}
        <Box sx={{ mb: 4 }}>
          <AgentStatus agents={agents} onRefresh={handleRefreshAgents} />
        </Box>

        {/* Quick Stats */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="primary">
                  $124.5K
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Revenue
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="secondary">
                  2,845
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Customers
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="success.main">
                  94%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Satisfaction
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" fontWeight="bold" color="warning.main">
                  98%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  System Uptime
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Main Tabs */}
        <Paper sx={{ background: 'rgba(30, 41, 59, 0.8)', backdropFilter: 'blur(10px)' }}>
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
              {React.cloneElement(tab.component, { data })}
            </TabPanel>
          ))}
        </Paper>
      </Container>
    </Box>
  );
};

export default Dashboard;