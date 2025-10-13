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

  const handleCalculateKPIs = async () => {
    setCalculating(true);
    setKpis(null);
    
    try {
      const response = await getKPIs();
      setKpis(response);
    } catch (error) {
      setKpis({ success: false, error: error.message });
    } finally {
      setCalculating(false);
    }
  };

  // Mock KPI data for demonstration
  const mockKpis = [
    {
      store_id: 'Los Angeles',
      metrics: {
        total_sales: 124560,
        sales_count: 1847,
        average_order_value: 67.45,
        total_items_sold: 5421
      },
      by_customer_category: {
        'VIP': 45600,
        'Regular': 65400,
        'New': 13560
      },
      by_payment_method: {
        'Credit Card': 78900,
        'Debit Card': 32400,
        'Cash': 8760,
        'Digital Wallet': 4500
      }
    },
    {
      store_id: 'New York',
      metrics: {
        total_sales: 98760,
        sales_count: 1521,
        average_order_value: 64.92,
        total_items_sold: 4215
      }
    }
  ];

  const displayKpis = kpis?.success ? kpis.data : mockKpis;

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

      {kpis && (
        <Box>
          {kpis.success ? (
            <Alert severity="success" sx={{ mb: 3 }}>
              ✅ KPIs calculated successfully!
            </Alert>
          ) : (
            <Alert severity="warning" sx={{ mb: 3 }}>
              Using demo KPI data. {kpis.error}
            </Alert>
          )}
        </Box>
      )}

      {displayKpis && displayKpis.length > 0 ? (
        <Box>
          <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
            📈 Store Performance KPIs
          </Typography>

          {displayKpis.map((kpi, index) => (
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
                        ${kpi.metrics?.total_sales?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Total Sales</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="secondary">
                        {kpi.metrics?.sales_count?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Transactions</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        ${kpi.metrics?.average_order_value?.toFixed(2) || '0.00'}
                      </Typography>
                      <Typography variant="body2">Avg Order Value</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="warning.main">
                        {kpi.metrics?.total_items_sold?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Items Sold</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Customer Category Breakdown */}
              {kpi.by_customer_category && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Customer Category Breakdown
                  </Typography>
                  <Grid container spacing={1}>
                    {Object.entries(kpi.by_customer_category).map(([category, amount]) => (
                      <Grid item xs={6} sm={4} key={category}>
                        <Card variant="outlined">
                          <CardContent sx={{ textAlign: 'center', py: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {category}
                            </Typography>
                            <Typography variant="body2" color="primary">
                              ${amount.toLocaleString()}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {/* Payment Method Breakdown */}
              {kpi.by_payment_method && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Payment Method Breakdown
                  </Typography>
                  <Grid container spacing={1}>
                    {Object.entries(kpi.by_payment_method).map(([method, amount]) => (
                      <Grid item xs={6} sm={3} key={method}>
                        <Card variant="outlined">
                          <CardContent sx={{ textAlign: 'center', py: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {method}
                            </Typography>
                            <Typography variant="body2" color="secondary">
                              ${amount.toLocaleString()}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {index < displayKpis.length - 1 && <Divider sx={{ mt: 3 }} />}
            </Paper>
          ))}
        </Box>
      ) : (
        <Alert severity="info">
          No KPI data available. Click "Calculate KPIs" to generate performance metrics.
        </Alert>
      )}
    </Box>
  );
};

export default KpiTab;