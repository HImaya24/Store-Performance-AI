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
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Store as StoreIcon,
  AttachMoney as MoneyIcon,
  ShoppingCart as CartIcon,
} from '@mui/icons-material';
import { semanticSearch } from '../../services/api';

const IRSearchTab = () => {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState(null);

  const exampleQueries = {
    "Customer Behavior": [
      "loyal customers with repeat purchases",
      "high value transactions", 
      "new customer acquisition patterns"
    ],
    "Seasonal Trends": [
      "winter season best sellers",
      "holiday shopping patterns",
      "seasonal product preferences"
    ],
    "Payment & Store Patterns": [
      "mobile payment adoption",
      "premium store customers", 
      "discount-driven purchases"
    ]
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setSearching(true);
    setResults(null);
    
    try {
      const searchResults = await semanticSearch(query);
      setResults(searchResults);
    } catch (error) {
      setResults({ error: 'Search failed', results: [] });
    } finally {
      setSearching(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        🎯 IR Search (Semantic Search)
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>IR Search</strong> uses semantic understanding to find patterns in your data. 
        Instead of just matching keywords, it understands the meaning of your queries.
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                fullWidth
                label="Enter your semantic search query"
                placeholder="e.g., high value customers, winter sales trends, mobile payment users..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                disabled={searching}
              />
              <Button
                variant="contained"
                startIcon={searching ? <CircularProgress size={20} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={searching || !query.trim()}
                sx={{ minWidth: 140 }}
              >
                {searching ? 'Searching...' : 'Run IR Search'}
              </Button>
            </Box>

            {results && (
              <Box>
                {results.results && results.results.length > 0 ? (
                  <Box>
                    <Alert severity="success" sx={{ mb: 2 }}>
                      ✅ Found {results.results.length} semantic matches for '{results.query}'
                    </Alert>
                    {results.results.map((result, index) => (
                      <Card key={index} variant="outlined" sx={{ mb: 2, borderLeft: 4, borderLeftColor: 'primary.main' }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                            <Typography variant="h6">
                              Match #{index + 1}
                            </Typography>
                            <Chip 
                              label={`Score: ${result.similarity_score?.toFixed(2) || '0.00'}`}
                              color="success"
                              size="small"
                            />
                          </Box>
                          
                          <Grid container spacing={2}>
                            <Grid item xs={12} sm={6}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <StoreIcon fontSize="small" color="action" />
                                <Typography variant="body2">
                                  <strong>Store:</strong> {result.metadata?.store_id || 'Unknown'}
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <MoneyIcon fontSize="small" color="action" />
                                <Typography variant="body2">
                                  <strong>Amount:</strong> ${result.metadata?.amount?.toFixed(2) || '0.00'}
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <CartIcon fontSize="small" color="action" />
                                <Typography variant="body2">
                                  <strong>Products:</strong> {result.metadata?.products?.join(', ') || 'None'}
                                </Typography>
                              </Box>
                            </Grid>
                          </Grid>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            <strong>Preview:</strong> {result.document_preview || 'No preview available'}
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                ) : (
                  <Alert severity="warning">
                    ❌ No semantic matches found for '{results.query}'
                  </Alert>
                )}
                
                {results.error && !results.results && (
                  <Alert severity="error">
                    ❌ IR Search failed: {results.error}
                  </Alert>
                )}
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              💡 Example Queries
            </Typography>
            
            {Object.entries(exampleQueries).map(([category, queries]) => (
              <Accordion key={category} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2" fontWeight="bold">
                    {category}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {queries.map((example, index) => (
                    <Chip
                      key={index}
                      label={example}
                      variant="outlined"
                      size="small"
                      onClick={() => setQuery(example)}
                      sx={{ m: 0.5, cursor: 'pointer' }}
                    />
                  ))}
                </AccordionDetails>
              </Accordion>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default IRSearchTab;