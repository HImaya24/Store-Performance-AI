import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
  CircularProgress,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

const ProductSearchTab = ({ data }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);

  const extractUniqueProducts = (events) => {
    const products = new Set();
    events?.forEach(event => {
      const items = event.payload?.items;
      if (items) {
        if (Array.isArray(items)) {
          items.forEach(item => products.add(item));
        } else {
          products.add(items);
        }
      }
    });
    return Array.from(products).sort();
  };

  const searchProducts = (query) => {
    if (!data?.data) return [];
    
    const results = [];
    const queryLower = query.toLowerCase();
    
    data.data.forEach(event => {
      const items = event.payload?.items;
      if (items) {
        const itemList = Array.isArray(items) ? items : [items];
        itemList.forEach(item => {
          if (item.toLowerCase().includes(queryLower)) {
            results.push({
              event_id: event.event_id,
              store_id: event.store_id,
              timestamp: event.ts,
              product: item,
              amount: event.payload?.amount || 0,
              customer_category: event.payload?.customer_category || 'Unknown',
              payment_method: event.payload?.payment_method || 'Unknown',
              season: event.payload?.season || 'Unknown',
            });
          }
        });
      }
    });
    
    return results;
  };

  const createProductCharts = (searchResults, productName) => {
    const charts = {};
    
    if (!searchResults || searchResults.length === 0) {
      return charts;
    }

    // Use ACTUAL search results data for charts
    const results = searchResults;

    // 1. Sales Over Time - Using actual timestamps from search results
    const monthlySales = {};
    results.forEach(result => {
      const month = new Date(result.timestamp).toLocaleDateString('en-US', { 
        month: 'short', 
        year: 'numeric' 
      });
      monthlySales[month] = (monthlySales[month] || 0) + result.amount;
    });
    
    charts.salesTrend = Object.entries(monthlySales)
      .map(([month, amount]) => ({
        month,
        amount: Math.round(amount)
      }))
      .sort((a, b) => new Date(a.month) - new Date(b.month));

    // 2. Sales by Store - Using actual store data from search results
    const storeSales = {};
    results.forEach(result => {
      storeSales[result.store_id] = (storeSales[result.store_id] || 0) + result.amount;
    });
    
    charts.storeSales = Object.entries(storeSales)
      .map(([store, amount]) => ({
        store,
        amount: Math.round(amount)
      }))
      .sort((a, b) => b.amount - a.amount);

    // 3. Sales by Season - Using actual season data from search results
    const seasonSales = {};
    results.forEach(result => {
      const season = result.season || 'Unknown';
      seasonSales[season] = (seasonSales[season] || 0) + result.amount;
    });
    
    charts.seasonSales = Object.entries(seasonSales)
      .map(([season, amount]) => ({
        season,
        amount: Math.round(amount),
        fill: getSeasonColor(season)
      }))
      .sort((a, b) => b.amount - a.amount);

    // 4. Customer Category Analysis - Using actual customer data from search results
    const customerSales = {};
    results.forEach(result => {
      const category = result.customer_category || 'Unknown';
      customerSales[category] = (customerSales[category] || 0) + result.amount;
    });
    
    charts.customerSales = Object.entries(customerSales)
      .map(([category, amount]) => ({
        category,
        amount: Math.round(amount)
      }))
      .sort((a, b) => b.amount - a.amount);

    return charts;
  };

  const getSeasonColor = (season) => {
    const colors = {
      'winter': '#8884d8',
      'spring': '#82ca9d', 
      'summer': '#ffc658',
      'fall': '#ff7300',
      'Unknown': '#8dd1e1'
    };
    return colors[season] || '#8884d8';
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setSearching(true);
    // Simulate API call delay
    setTimeout(() => {
      const results = searchProducts(searchQuery);
      setSearchResults(results);
      setSearching(false);
    }, 1000);
  };

  const uniqueProducts = data?.data ? extractUniqueProducts(data.data) : [];
  const productCharts = searchResults ? createProductCharts(searchResults, searchQuery) : {};
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        🔎 Product Search & Analysis
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                fullWidth
                label="🔍 Search for products"
                placeholder="Enter product name (e.g., butter, apple, olive oil...)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                disabled={searching}
              />
              <Button
                variant="contained"
                startIcon={searching ? <CircularProgress size={20} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={searching || !searchQuery.trim()}
                sx={{ minWidth: 140 }}
              >
                {searching ? 'Searching...' : 'Analyze Product'}
              </Button>
            </Box>

            {uniqueProducts.length > 0 && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <strong>Available products:</strong> {uniqueProducts.slice(0, 10).join(', ')}
                {uniqueProducts.length > 10 && '...'}
              </Alert>
            )}

            {searchResults && (
              <Box>
                {searchResults.length > 0 ? (
                  <Box>
                    <Alert severity="success" sx={{ mb: 2 }}>
                      ✅ Found {searchResults.length} transactions for products matching '{searchQuery}'
                    </Alert>

                    {/* Summary Metrics - Using REAL data */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6} sm={3}>
                        <Card>
                          <CardContent sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="primary">
                              ${searchResults.reduce((sum, r) => sum + r.amount, 0).toFixed(2)}
                            </Typography>
                            <Typography variant="body2">Total Revenue</Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Card>
                          <CardContent sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="secondary">
                              {searchResults.length}
                            </Typography>
                            <Typography variant="body2">Transactions</Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Card>
                          <CardContent sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="success.main">
                              {new Set(searchResults.map(r => r.store_id)).size}
                            </Typography>
                            <Typography variant="body2">Stores</Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Card>
                          <CardContent sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="warning.main">
                              ${(searchResults.reduce((sum, r) => sum + r.amount, 0) / searchResults.length).toFixed(2)}
                            </Typography>
                            <Typography variant="body2">Avg Sale</Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>

                    {/* Product Insights */}
                    <Card sx={{ mb: 2, borderLeft: 4, borderLeftColor: 'success.main' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          📈 Product Insights
                        </Typography>
                        <Typography><strong>Product:</strong> {searchQuery}</Typography>
                        <Typography><strong>Total Revenue:</strong> ${searchResults.reduce((sum, r) => sum + r.amount, 0).toFixed(2)}</Typography>
                        <Typography><strong>Total Transactions:</strong> {searchResults.length}</Typography>
                        <Typography><strong>Stores:</strong> {Array.from(new Set(searchResults.map(r => r.store_id))).join(', ')}</Typography>
                      </CardContent>
                    </Card>

                    {/* PRODUCT CHARTS - Using REAL data */}
                    <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                      📊 Product Performance Charts
                    </Typography>

                    <Grid container spacing={3}>
                      {/* Sales Trend Chart */}
                      {productCharts.salesTrend && productCharts.salesTrend.length > 0 && (
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle1" gutterBottom align="center">
                              Monthly Sales Trend for {searchQuery}
                            </Typography>
                            <ResponsiveContainer width="100%" height={250}>
                              <LineChart data={productCharts.salesTrend}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="month" />
                                <YAxis />
                                <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                <Line type="monotone" dataKey="amount" stroke="#8884d8" strokeWidth={2} />
                              </LineChart>
                            </ResponsiveContainer>
                          </Paper>
                        </Grid>
                      )}

                      {/* Sales by Store Chart */}
                      {productCharts.storeSales && productCharts.storeSales.length > 0 && (
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle1" gutterBottom align="center">
                              Sales by Store for {searchQuery}
                            </Typography>
                            <ResponsiveContainer width="100%" height={250}>
                              <BarChart data={productCharts.storeSales}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="store" />
                                <YAxis />
                                <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                <Bar dataKey="amount" fill="#82ca9d" />
                              </BarChart>
                            </ResponsiveContainer>
                          </Paper>
                        </Grid>
                      )}

                      {/* Sales by Season Chart */}
                      {productCharts.seasonSales && productCharts.seasonSales.length > 0 && (
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle1" gutterBottom align="center">
                              Sales Distribution by Season for {searchQuery}
                            </Typography>
                            <ResponsiveContainer width="100%" height={250}>
                              <PieChart>
                                <Pie
                                  data={productCharts.seasonSales}
                                  cx="50%"
                                  cy="50%"
                                  labelLine={false}
                                  label={({ season, percent }) => `${season} ${(percent * 100).toFixed(0)}%`}
                                  outerRadius={80}
                                  fill="#8884d8"
                                  dataKey="amount"
                                >
                                  {productCharts.seasonSales.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.fill || colors[index % colors.length]} />
                                  ))}
                                </Pie>
                                <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                              </PieChart>
                            </ResponsiveContainer>
                          </Paper>
                        </Grid>
                      )}

                      {/* Customer Category Chart */}
                      {productCharts.customerSales && productCharts.customerSales.length > 0 && (
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle1" gutterBottom align="center">
                              Sales by Customer Category for {searchQuery}
                            </Typography>
                            <ResponsiveContainer width="100%" height={250}>
                              <BarChart data={productCharts.customerSales}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="category" />
                                <YAxis />
                                <Tooltip formatter={(value) => [`$${value}`, 'Sales']} />
                                <Bar dataKey="amount" fill="#ffc658" />
                              </BarChart>
                            </ResponsiveContainer>
                          </Paper>
                        </Grid>
                      )}
                    </Grid>

                    {/* Sample Results */}
                    <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                      Sample Transactions
                    </Typography>
                    {searchResults.slice(0, 5).map((result, index) => (
                      <Card key={index} sx={{ mb: 1 }}>
                        <CardContent>
                          <Grid container spacing={2}>
                            <Grid item xs={8}>
                              <Typography variant="body2"><strong>Store:</strong> {result.store_id}</Typography>
                              <Typography variant="body2"><strong>Amount:</strong> ${result.amount.toFixed(2)}</Typography>
                              <Typography variant="body2"><strong>Customer:</strong> {result.customer_category}</Typography>
                            </Grid>
                            <Grid item xs={4}>
                              <Chip label={result.season} size="small" />
                              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                                {new Date(result.timestamp).toLocaleDateString()}
                              </Typography>
                            </Grid>
                          </Grid>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                ) : (
                  <Alert severity="warning">
                    ❌ No transactions found for products matching '{searchQuery}'
                  </Alert>
                )}
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              💡 Search Tips
            </Typography>
            <Typography variant="body2" paragraph>
              Try searching for common products like:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {['bread', 'baby wipes', 'olive oil', 'microwave', 'beef'].map(product => (
                <Chip
                  key={product}
                  label={product}
                  variant="outlined"
                  onClick={() => setSearchQuery(product)}
                  clickable
                />
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ProductSearchTab;