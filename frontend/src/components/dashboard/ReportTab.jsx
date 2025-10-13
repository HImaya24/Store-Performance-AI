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
import { generateReport } from '../../services/api';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

const ReportTab = ({ data }) => {
  const [generating, setGenerating] = useState(false);
  const [selectedStore, setSelectedStore] = useState('');
  const [report, setReport] = useState(null);

  // Extract unique stores from data
  const availableStores = data?.data 
    ? [...new Set(data.data.map(item => item.store_id))].sort()
    : ['Los Angeles', 'New York', 'Chicago', 'Miami', 'Seattle'];

  // Create report charts data using REAL data
  const createReportCharts = (storeId) => {
    if (!data?.data) return {};
    
    // Use ACTUAL events data for the selected store
    const storeEvents = data.data.filter(event => event.store_id === storeId);
    const charts = {};

    // 1. Sales Over Time - Using actual sales data
    const salesOverTime = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.amount) {
        const date = new Date(event.ts).toLocaleDateString();
        salesOverTime[date] = (salesOverTime[date] || 0) + event.payload.amount;
      }
    });
    
    charts.salesOverTime = Object.entries(salesOverTime)
      .map(([date, amount]) => ({
        date,
        amount: Math.round(amount)
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .slice(-10); // Show last 10 dates

    // 2. Event Type Distribution - Using actual event types
    const eventDistribution = {};
    storeEvents.forEach(event => {
      const type = event.event_type;
      eventDistribution[type] = (eventDistribution[type] || 0) + 1;
    });
    
    charts.eventDistribution = Object.entries(eventDistribution)
      .map(([name, value]) => ({
        name,
        value
      }))
      .sort((a, b) => b.value - a.value);

    // 3. Monthly Sales Trend - Using actual monthly data
    const monthlySales = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.amount) {
        const month = new Date(event.ts).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        monthlySales[month] = (monthlySales[month] || 0) + event.payload.amount;
      }
    });
    
    charts.monthlySales = Object.entries(monthlySales)
      .map(([month, sales]) => ({
        month,
        sales: Math.round(sales)
      }))
      .sort((a, b) => new Date(a.month) - new Date(b.month));

    // 4. Payment Method Analysis - Using actual payment data
    const paymentMethods = {};
    storeEvents.forEach(event => {
      if (event.event_type === 'sale' && event.payload?.payment_method) {
        const method = event.payload.payment_method;
        paymentMethods[method] = (paymentMethods[method] || 0) + (event.payload.amount || 0);
      }
    });
    
    charts.paymentMethods = Object.entries(paymentMethods)
      .map(([method, amount]) => ({
        method,
        amount: Math.round(amount)
      }))
      .sort((a, b) => b.amount - a.amount);

    return charts;
  };

  const handleGenerateReport = async () => {
    if (!selectedStore) return;

    setGenerating(true);
    setReport(null);
    
    try {
      const response = await generateReport(selectedStore);
      setReport(response);
    } catch (error) {
      setReport({ success: false, error: error.message });
    } finally {
      setGenerating(false);
    }
  };

  // Generate comprehensive HTML report with REAL data
  const generateHTMLReport = (storeName, charts, storeEvents) => {
    // Calculate REAL metrics from store data
    const storeSales = storeEvents.filter(e => e.event_type === 'sale' && e.payload?.amount);
    const totalSales = storeSales.reduce((sum, e) => sum + (e.payload.amount || 0), 0);
    const transactionCount = storeSales.length;
    const avgOrderValue = transactionCount > 0 ? totalSales / transactionCount : 0;
    const totalItems = storeSales.reduce((sum, e) => {
      const items = e.payload?.items;
      if (Array.isArray(items)) return sum + items.length;
      return sum + 1;
    }, 0);

    // Count events by type
    const eventCounts = {};
    storeEvents.forEach(event => {
      eventCounts[event.event_type] = (eventCounts[event.event_type] || 0) + 1;
    });

    return `
<!DOCTYPE html>
<html>
<head>
    <title>Store Performance Report - ${storeName}</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section-title {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #3498db;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .chart-container {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .chart-title {
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: #3498db;
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9em;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .event-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .event-type {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .event-count {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .event-name {
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .footer {
            text-align: center;
            padding: 30px;
            background: #2c3e50;
            color: white;
            margin-top: 40px;
        }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <h1>Store Performance Report</h1>
            <div class="subtitle">${storeName}</div>
            <div class="subtitle">Generated on: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}</div>
        </div>

        <div class="content">
            <!-- Key Metrics Section -->
            <div class="section">
                <h2 class="section-title">📊 Key Performance Indicators</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">$${totalSales.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                        <div class="metric-label">Total Revenue</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${transactionCount}</div>
                        <div class="metric-label">Total Transactions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">$${avgOrderValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                        <div class="metric-label">Average Order Value</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${totalItems}</div>
                        <div class="metric-label">Items Sold</div>
                    </div>
                </div>
            </div>

            <!-- Event Breakdown -->
            <div class="section">
                <h2 class="section-title">📈 Activity Overview</h2>
                <div class="event-breakdown">
                    ${Object.entries(eventCounts).map(([eventType, count]) => `
                        <div class="event-type">
                            <div class="event-count">${count}</div>
                            <div class="event-name">${eventType} Events</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Sales Over Time -->
            ${charts.salesOverTime && charts.salesOverTime.length > 0 ? `
            <div class="section">
                <h2 class="section-title">💰 Sales Over Time</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Sales Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${charts.salesOverTime.map(item => `
                            <tr>
                                <td>${item.date}</td>
                                <td>$${item.amount.toLocaleString()}</td>
                            </tr>
                        `).join('')}
                        <tr style="background: #e8f5e8; font-weight: bold;">
                            <td>Total</td>
                            <td>$${charts.salesOverTime.reduce((sum, item) => sum + item.amount, 0).toLocaleString()}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            ` : '<div class="no-data">No sales data available for the selected period</div>'}

            <!-- Monthly Sales -->
            ${charts.monthlySales && charts.monthlySales.length > 0 ? `
            <div class="section">
                <h2 class="section-title">📅 Monthly Sales Trend</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Month</th>
                            <th>Sales Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${charts.monthlySales.map(item => `
                            <tr>
                                <td>${item.month}</td>
                                <td>$${item.sales.toLocaleString()}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ` : ''}

            <!-- Payment Methods -->
            ${charts.paymentMethods && charts.paymentMethods.length > 0 ? `
            <div class="section">
                <h2 class="section-title">💳 Payment Method Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Payment Method</th>
                            <th>Total Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${charts.paymentMethods.map(item => `
                            <tr>
                                <td>${item.method}</td>
                                <td>$${item.amount.toLocaleString()}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ` : ''}

            <!-- Event Distribution -->
            ${charts.eventDistribution && charts.eventDistribution.length > 0 ? `
            <div class="section">
                <h2 class="section-title">🔔 Event Type Distribution</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Event Type</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${charts.eventDistribution.map(item => `
                            <tr>
                                <td>${item.name}</td>
                                <td>${item.value}</td>
                            </tr>
                        `).join('')}
                        <tr style="background: #e3f2fd; font-weight: bold;">
                            <td>Total Events</td>
                            <td>${charts.eventDistribution.reduce((sum, item) => sum + item.value, 0)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            ` : ''}
        </div>

        <div class="footer">
            <p>Generated by Store Analytics Dashboard | ${storeName} Performance Report</p>
            <p>© ${new Date().getFullYear()} All rights reserved</p>
        </div>
    </div>
</body>
</html>`;
  };

  const handleDownloadReport = () => {
    if (!selectedStore) return;

    const reportCharts = createReportCharts(selectedStore);
    const storeEvents = data?.data ? data.data.filter(event => event.store_id === selectedStore) : [];
    
    // Generate HTML report with REAL data
    const htmlContent = generateHTMLReport(selectedStore, reportCharts, storeEvents);
    
    // Create and download the blob with ACTUAL content
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
                          <Typography variant="h6" gutterBottom>
                            Store Performance Report - {selectedStore}
                          </Typography>
                          
                          {/* REPORT CHARTS - Using REAL data */}
                          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                            📊 Store Performance Charts
                          </Typography>

                          <Grid container spacing={2}>
                            {/* Sales Over Time Chart */}
                            {reportCharts.salesOverTime && reportCharts.salesOverTime.length > 0 && (
                              <Grid item xs={12}>
                                <Paper sx={{ p: 2, mb: 2 }}>
                                  <Typography variant="subtitle1" gutterBottom align="center">
                                    Sales Over Time - {selectedStore}
                                  </Typography>
                                  <ResponsiveContainer width="100%" height={250}>
                                    <LineChart data={reportCharts.salesOverTime}>
                                      <CartesianGrid strokeDasharray="3 3" />
                                      <XAxis dataKey="date" />
                                      <YAxis />
                                      <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                      <Line type="monotone" dataKey="amount" stroke="#8884d8" strokeWidth={2} />
                                    </LineChart>
                                  </ResponsiveContainer>
                                </Paper>
                              </Grid>
                            )}

                            {/* Event Distribution Chart */}
                            {reportCharts.eventDistribution && reportCharts.eventDistribution.length > 0 && (
                              <Grid item xs={12} md={6}>
                                <Paper sx={{ p: 2, mb: 2 }}>
                                  <Typography variant="subtitle1" gutterBottom align="center">
                                    Event Type Distribution - {selectedStore}
                                  </Typography>
                                  <ResponsiveContainer width="100%" height={250}>
                                    <PieChart>
                                      <Pie
                                        data={reportCharts.eventDistribution}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                      >
                                        {reportCharts.eventDistribution.map((entry, index) => (
                                          <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                                        ))}
                                      </Pie>
                                      <Tooltip />
                                    </PieChart>
                                  </ResponsiveContainer>
                                </Paper>
                              </Grid>
                            )}

                            {/* Monthly Sales Chart */}
                            {reportCharts.monthlySales && reportCharts.monthlySales.length > 0 && (
                              <Grid item xs={12} md={6}>
                                <Paper sx={{ p: 2, mb: 2 }}>
                                  <Typography variant="subtitle1" gutterBottom align="center">
                                    Monthly Sales Trend - {selectedStore}
                                  </Typography>
                                  <ResponsiveContainer width="100%" height={250}>
                                    <BarChart data={reportCharts.monthlySales}>
                                      <CartesianGrid strokeDasharray="3 3" />
                                      <XAxis dataKey="month" />
                                      <YAxis />
                                      <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                      <Bar dataKey="sales" fill="#82ca9d" />
                                    </BarChart>
                                  </ResponsiveContainer>
                                </Paper>
                              </Grid>
                            )}

                            {/* Payment Method Chart */}
                            {reportCharts.paymentMethods && reportCharts.paymentMethods.length > 0 && (
                              <Grid item xs={12} md={6}>
                                <Paper sx={{ p: 2, mb: 2 }}>
                                  <Typography variant="subtitle1" gutterBottom align="center">
                                    Payment Methods - {selectedStore}
                                  </Typography>
                                  <ResponsiveContainer width="100%" height={250}>
                                    <BarChart data={reportCharts.paymentMethods}>
                                      <CartesianGrid strokeDasharray="3 3" />
                                      <XAxis dataKey="method" />
                                      <YAxis />
                                      <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                      <Bar dataKey="amount" fill="#ffc658" />
                                    </BarChart>
                                  </ResponsiveContainer>
                                </Paper>
                              </Grid>
                            )}
                          </Grid>

                          {/* Store Summary with REAL data */}
                          <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                              Store Performance Summary - {selectedStore}
                            </Typography>
                            
                            {(() => {
                              // Calculate REAL metrics from store data
                              const storeSales = storeEvents.filter(e => e.event_type === 'sale' && e.payload?.amount);
                              const totalSales = storeSales.reduce((sum, e) => sum + (e.payload.amount || 0), 0);
                              const transactionCount = storeSales.length;
                              const avgOrderValue = transactionCount > 0 ? totalSales / transactionCount : 0;
                              const totalItems = storeSales.reduce((sum, e) => {
                                const items = e.payload?.items;
                                if (Array.isArray(items)) return sum + items.length;
                                return sum + 1;
                              }, 0);

                              return (
                                <Grid container spacing={2}>
                                  <Grid item xs={6} sm={3}>
                                    <Card>
                                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                                        <Typography variant="h6" color="primary">
                                          ${totalSales.toFixed(2)}
                                        </Typography>
                                        <Typography variant="body2">Total Sales</Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                  <Grid item xs={6} sm={3}>
                                    <Card>
                                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                                        <Typography variant="h6" color="secondary">
                                          {transactionCount}
                                        </Typography>
                                        <Typography variant="body2">Transactions</Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                  <Grid item xs={6} sm={3}>
                                    <Card>
                                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                                        <Typography variant="h6" color="success.main">
                                          ${avgOrderValue.toFixed(2)}
                                        </Typography>
                                        <Typography variant="body2">Avg Order Value</Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                  <Grid item xs={6} sm={3}>
                                    <Card>
                                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                                        <Typography variant="h6" color="warning.main">
                                          {totalItems}
                                        </Typography>
                                        <Typography variant="body2">Items Sold</Typography>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                </Grid>
                              );
                            })()}
                          </Box>
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