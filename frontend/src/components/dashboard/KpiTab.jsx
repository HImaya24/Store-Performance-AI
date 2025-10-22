import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Divider,
} from '@mui/material';
import { Calculate as CalculateIcon } from '@mui/icons-material';
import { getKPIs } from '../../services/api';

const KpiTab = () => {
  const [calculating, setCalculating] = useState(false);
  const [kpis, setKpis] = useState(null);
  const [error, setError] = useState(null);

  const handleCalculateKPIs = async () => {
    setCalculating(true);
    setKpis(null);
    setError(null);
    
    try {
      const kpiData = await getKPIs();
      console.log('📊 KPI Data received:', kpiData);
      
      if (Array.isArray(kpiData) && kpiData.length > 0) {
        setKpis(kpiData);
        setError(null);
      } else {
        setError('No KPI data received from server');
        setKpis([]);
      }
    } catch (error) {
      console.error('KPI calculation error:', error);
      setError(error.message);
      setKpis([]);
    } finally {
      setCalculating(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        📈 KPI Agent
      </Typography>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button
          variant="contained"
          startIcon={calculating ? <CircularProgress size={20} /> : <CalculateIcon />}
          onClick={handleCalculateKPIs}
          disabled={calculating}
        >
          {calculating ? 'Calculating...' : 'Calculate KPIs'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          ❌ Error: {error}
        </Alert>
      )}

      {kpis && kpis.length > 0 && (
        <Alert severity="success" sx={{ mb: 3 }}>
          ✅ KPIs calculated successfully! Found {kpis.length} stores with real data.
        </Alert>
      )}

      {kpis && kpis.length > 0 ? (
        <Box>
          <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
            📈 Store Performance KPIs
          </Typography>

          {kpis.map((kpi, index) => (
            <Paper key={index} sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom color="primary">
                🏪 {kpi.store_id || 'Unknown Store'}
              </Typography>

              {/* Main Metrics */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        ${kpi.metrics?.total_sales?.toLocaleString() || kpi.total_sales?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Total Sales</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="secondary">
                        {kpi.metrics?.sales_count?.toLocaleString() || kpi.sales_count?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Transactions</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        ${(kpi.metrics?.average_order_value || kpi.average_order_value || 0).toFixed(2)}
                      </Typography>
                      <Typography variant="body2">Avg Order Value</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="warning.main">
                        {kpi.metrics?.total_items_sold?.toLocaleString() || kpi.total_items_sold?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Items Sold</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Customer Category Breakdown */}
              {(kpi.by_customer_category || kpi.metrics?.by_customer_category) && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    👥 Customer Category Breakdown
                  </Typography>
                  <Grid container spacing={1}>
                    {Object.entries(kpi.by_customer_category || kpi.metrics?.by_customer_category || {}).map(([category, amount]) => (
                      <Grid item xs={6} sm={4} key={category}>
                        <Card variant="outlined">
                          <CardContent sx={{ textAlign: 'center', py: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {category}
                            </Typography>
                            <Typography variant="body2" color="primary">
                              ${typeof amount === 'number' ? amount.toLocaleString() : amount}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {/* Payment Method Breakdown */}
              {(kpi.by_payment_method || kpi.metrics?.by_payment_method) && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    💳 Payment Method Breakdown
                  </Typography>
                  <Grid container spacing={1}>
                    {Object.entries(kpi.by_payment_method || kpi.metrics?.by_payment_method || {}).map(([method, amount]) => (
                      <Grid item xs={6} sm={3} key={method}>
                        <Card variant="outlined">
                          <CardContent sx={{ textAlign: 'center', py: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {method}
                            </Typography>
                            <Typography variant="body2" color="secondary">
                              ${typeof amount === 'number' ? amount.toLocaleString() : amount}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {index < kpis.length - 1 && <Divider sx={{ mt: 3 }} />}
            </Paper>
          ))}
        </Box>
      ) : (
        !calculating && (
          <Alert severity="info">
            📊 No KPI data available. Click "Calculate KPIs" to generate performance metrics from your real data.
          </Alert>
        )
      )}

      {calculating && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="h6">Calculating REAL KPIs...</Typography>
          <Typography variant="body2" color="text.secondary">
            Processing your  transaction data
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default KpiTab;