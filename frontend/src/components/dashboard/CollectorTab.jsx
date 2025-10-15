import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { loadDataFromCollector } from '../../services/api';

const CollectorTab = ({ data, onDataUpdate }) => {
  const handleRefresh = async () => {
    if (onDataUpdate) {
      const newData = await loadDataFromCollector();
      onDataUpdate(newData);
    }
  };

  if (!data) {
    return (
      <Alert severity="info">
        No data available. Please check if the Collector agent is running.
      </Alert>
    );
  }

  const { data: events, fromCollector } = data;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="bold">
          📥 Collector Agent
        </Typography>
        <Button
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          variant="outlined"
        >
          Refresh Data
        </Button>
      </Box>

      <Alert 
        severity={fromCollector ? "success" : "warning"}
        sx={{ mb: 3 }}
      >
        {fromCollector 
          ? "✅ Data loaded successfully from Collector agent (port 8100)"
          : "⚠️ Using sample data (Collector agent on port 8100 unavailable)"
        }
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Data Preview
            </Typography>
            <TableContainer sx={{ maxHeight: 400 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Event ID</TableCell>
                    <TableCell>Store</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {events.slice(0, 10).map((row, index) => (
                    <TableRow key={index}>
                      <TableCell>{row.event_id}</TableCell>
                      <TableCell>{row.store_id}</TableCell>
                      <TableCell>{row.event_type}</TableCell>
                      <TableCell>${row.payload?.amount?.toFixed(2)}</TableCell>
                      <TableCell>{new Date(row.ts).toLocaleDateString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Data Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    {events?.length || 0}
                  </Typography>
                  <Typography variant="body2">Total Events</Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'secondary.light', borderRadius: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    {new Set(events?.map(d => d.store_id)).size || 0}
                  </Typography>
                  <Typography variant="body2">Unique Stores</Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    {new Set(events?.map(d => d.event_type)).size || 0}
                  </Typography>
                  <Typography variant="body2">Event Types</Typography>
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
                  <Typography variant="h4" fontWeight="bold">
                    ${events?.reduce((sum, d) => sum + (d.payload?.amount || 0), 0).toFixed(2)}
                  </Typography>
                  <Typography variant="body2">Total Revenue</Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CollectorTab;