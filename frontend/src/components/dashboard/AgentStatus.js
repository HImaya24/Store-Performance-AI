import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as OnlineIcon,
  Error as OfflineIcon,
} from '@mui/icons-material';

const AgentStatus = ({ agents, onRefresh }) => {
  const agentConfig = [
    { id: 'collector', name: 'Data Collector', port: 8100 },
    { id: 'coordinator', name: 'Coordinator', port: 8110 },
    { id: 'analyzer', name: 'AI Analyzer', port: 8101 },
    { id: 'kpi', name: 'KPI Engine', port: 8102 },
    { id: 'report', name: 'Report Generator', port: 8103 },
  ];

  const handleRefresh = async () => {
    if (onRefresh) {
      onRefresh();
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          🤖 Agent Status
        </Typography>
        <IconButton onClick={handleRefresh} size="small">
          <RefreshIcon />
        </IconButton>
      </Box>

      <Grid container spacing={2}>
        {agentConfig.map((agent) => {
          const status = agents[agent.id]?.status || 'offline';
          const isOnline = status === 'online';
          
          return (
            <Grid item xs={12} sm={6} md={4} key={agent.id}>
              <Card 
                variant="outlined" 
                sx={{ 
                  borderColor: isOnline ? 'success.main' : 'error.main',
                  opacity: isOnline ? 1 : 0.7,
                  height: '100%'
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    {isOnline ? (
                      <OnlineIcon sx={{ color: 'success.main', mr: 1 }} />
                    ) : (
                      <OfflineIcon sx={{ color: 'error.main', mr: 1 }} />
                    )}
                    <Typography variant="h6" component="div">
                      {agent.name}
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Port: {agent.port}
                  </Typography>
                  
                  <Chip
                    label={isOnline ? 'Online' : 'Offline'}
                    color={isOnline ? 'success' : 'error'}
                    size="small"
                    variant="outlined"
                  />
                  
                  {agents[agent.id]?.lastChecked && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Last checked: {new Date(agents[agent.id].lastChecked).toLocaleTimeString()}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default AgentStatus;