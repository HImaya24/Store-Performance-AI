import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import { Description as ReportIcon, Download as DownloadIcon } from '@mui/icons-material';
import { generateReport, fetchReportSummary } from '../../services/api'; // ✅ import new API
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

const ReportTab = ({ data }) => {
  const [generating, setGenerating] = useState(false);
  const [selectedStore, setSelectedStore] = useState('');
  const [report, setReport] = useState(null);
  const [aiSummary, setAiSummary] = useState(null); // ✅ new state

  const availableStores = data?.data 
    ? [...new Set(data.data.map(item => item.store_id))].sort()
    : ['Los Angeles', 'New York', 'Chicago', 'Miami', 'Seattle'];

  const createReportCharts = (storeId) => {
    if (!data?.data) return {};
    const storeEvents = data.data.filter(event => event.store_id === storeId);
    const charts = {};

    const salesOverTime = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.amount) {
        const date = new Date(event.ts).toLocaleDateString();
        salesOverTime[date] = (salesOverTime[date] || 0) + event.payload.amount;
      }
    });
    charts.salesOverTime = Object.entries(salesOverTime)
      .map(([date, amount]) => ({ date, amount: Math.round(amount) }))
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .slice(-10);

    const eventDistribution = {};
    storeEvents.forEach(event => {
      const type = event.event_type;
      eventDistribution[type] = (eventDistribution[type] || 0) + 1;
    });
    charts.eventDistribution = Object.entries(eventDistribution)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);

    const monthlySales = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.amount) {
        const month = new Date(event.ts).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        monthlySales[month] = (monthlySales[month] || 0) + event.payload.amount;
      }
    });
    charts.monthlySales = Object.entries(monthlySales)
      .map(([month, sales]) => ({ month, sales: Math.round(sales) }))
      .sort((a, b) => new Date(a.month) - new Date(b.month));

    const paymentMethods = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.payment_method) {
        const method = event.payload.payment_method;
        paymentMethods[method] = (paymentMethods[method] || 0) + (event.payload.amount || 0);
      }
    });
    charts.paymentMethods = Object.entries(paymentMethods)
      .map(([method, amount]) => ({ method, amount: Math.round(amount) }))
      .sort((a, b) => b.amount - a.amount);

    return charts;
  };

  // ✅ Updated handleGenerateReport to fetch AI summary too
  const handleGenerateReport = async () => {
    if (!selectedStore) return;

    setGenerating(true);
    setReport(null);
    setAiSummary(null);

    try {
      const [reportResponse, summaryResponse] = await Promise.all([
        generateReport(selectedStore),
        fetchReportSummary(selectedStore),
      ]);

      setReport(reportResponse);
      setAiSummary(summaryResponse.ai_summary); // ✅ set AI summary
    } catch (error) {
      setReport({ success: false, error: error.message });
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadReport = () => {
    if (!selectedStore) return;

    const reportCharts = createReportCharts(selectedStore);
    const storeEvents = data?.data ? data.data.filter(event => event.store_id === selectedStore) : [];
    
    const htmlContent = generateHTMLReport(selectedStore, reportCharts, storeEvents);
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `store_report_${selectedStore}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const reportCharts = selectedStore ? createReportCharts(selectedStore) : {};
  const storeEvents = data?.data ? data.data.filter(event => event.store_id === selectedStore) : [];
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        📋 Report Agent
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Report Configuration
            </Typography>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Select Store</InputLabel>
              <Select
                value={selectedStore}
                label="Select Store"
                onChange={(e) => setSelectedStore(e.target.value)}
              >
                {availableStores.map(store => (
                  <MenuItem key={store} value={store}>
                    {store}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              variant="contained"
              startIcon={generating ? <CircularProgress size={20} /> : <ReportIcon />}
              onClick={handleGenerateReport}
              disabled={generating || !selectedStore}
              fullWidth
              size="large"
              sx={{ mb: 2 }}
            >
              {generating ? 'Generating...' : 'Generate Report'}
            </Button>

            {selectedStore && (
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleDownloadReport}
                fullWidth
                disabled={!selectedStore}
              >
                Download HTML Report
              </Button>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Report Output
            </Typography>

            {report && (
              <Box>
                {report.success ? (
                  <Box>
                    <Alert severity="success" sx={{ mb: 2 }}>
                      ✅ Report for {selectedStore} generated successfully!
                    </Alert>

                    <Box
                      sx={{
                        p: 2,
                        bgcolor: 'background.default',
                        borderRadius: 1,
                        border: 1,
                        borderColor: 'divider',
                        minHeight: 300,
                        maxHeight: 500,
                        overflow: 'auto',
                      }}
                    >
                      {report.data.report_html ? (
                        <div 
                          dangerouslySetInnerHTML={{ __html: report.data.report_html }}
                          style={{ color: 'white' }}
                        />
                      ) : (
                        <Box>
                          {/* ...existing charts and summary code... */}

                          {/* ✅ AI Summary Section */}
                          {aiSummary && (
                            <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                              <Typography variant="h6" gutterBottom>
                                🧠 AI Insights
                              </Typography>
                              <Typography
                                component="pre"
                                sx={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}
                              >
                                {aiSummary}
                              </Typography>
                            </Box>
                          )}
                        </Box>
                      )}
                    </Box>
                  </Box>
                ) : (
                  <Alert severity="error">
                    ❌ Failed to generate report: {report.error}
                  </Alert>
                )}
              </Box>
            )}

            {!report && !generating && (
              <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 8 }}>
                Select a store and click "Generate Report" to create a performance report
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ReportTab;
