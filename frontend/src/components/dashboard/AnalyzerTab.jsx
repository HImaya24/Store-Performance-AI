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
  Card,
  CardContent,
  Grid,
  Chip,
} from '@mui/material';
import { Analytics as AnalyticsIcon } from '@mui/icons-material';

const AnalyzerTab = () => {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisType, setAnalysisType] = useState('sales_analysis');
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleRunAnalysis = async () => {
    setAnalyzing(true);
    setAnalysisResult(null);
    
    try {
      // First get data from collector (EXACTLY like your Streamlit code)
      const collectorResponse = await fetch('http://localhost:8100/events');
      if (!collectorResponse.ok) {
        throw new Error(`Collector returned status ${collectorResponse.status}`);
      }
      
      let events = await collectorResponse.json();
      
      // Limit to reasonable number for testing (like your Streamlit code)
      if (events.length > 10) {
        events = events.slice(0, 10);
      }

      // Send to analyzer with EXACT format from your Streamlit code
      const response = await fetch('http://localhost:8101/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(events), // Send events array directly, not wrapped in object
      });
      
      if (response.ok) {
        const analysisData = await response.json();
        setAnalysisResult({ success: true, data: analysisData });
      } else {
        const errorText = await response.text();
        setAnalysisResult({ 
          success: false, 
          error: `Analyzer returned status ${response.status}: ${errorText}` 
        });
      }
    } catch (error) {
      setAnalysisResult({ success: false, error: error.message });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        🔍 Analyzer Agent
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Configuration
            </Typography>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Analysis Type</InputLabel>
              <Select
                value={analysisType}
                label="Analysis Type"
                onChange={(e) => setAnalysisType(e.target.value)}
              >
                <MenuItem value="sales_analysis">Sales Analysis</MenuItem>
                <MenuItem value="inventory_analysis">Inventory Analysis</MenuItem>
                <MenuItem value="customer_analysis">Customer Analysis</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mb: 2 }}>
              <strong>Note:</strong> Analysis type is for display only. The analyzer processes all event data.
            </Alert>

            <Button
              variant="contained"
              startIcon={analyzing ? <CircularProgress size={20} /> : <AnalyticsIcon />}
              onClick={handleRunAnalysis}
              disabled={analyzing}
              fullWidth
              size="large"
            >
              {analyzing ? 'Analyzing...' : 'Run Analysis'}
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, minHeight: 400 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>

            {analysisResult && (
              <Box>
                {analysisResult.success ? (
                  <Box>
                    <Alert severity="success" sx={{ mb: 2 }}>
                      ✅ Analysis completed successfully!
                    </Alert>

                    {analysisResult.data?.insights_list && analysisResult.data.insights_list.length > 0 ? (
                      <Box>
                        <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                          📊 Generated Insights ({analysisResult.data.insights_list.length} total)
                        </Typography>

                        {analysisResult.data.insights_list.map((insight, index) => (
                          <Card key={index} sx={{ mb: 2, borderLeft: 4, borderLeftColor: 'primary.main' }}>
                            <CardContent>
                              <Grid container spacing={2}>
                                <Grid item xs={9}>
                                  <Typography variant="h6" gutterBottom>
                                    Insight #{index + 1}: {insight.text || 'No text available'}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary" paragraph>
                                    {insight.explanation || 'No explanation available'}
                                  </Typography>
                                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                    <Chip 
                                      label={`Store: ${insight.store_id || 'Unknown'}`} 
                                      size="small" 
                                      variant="outlined" 
                                    />
                                    {insight.tags?.map((tag, tagIndex) => (
                                      <Chip key={tagIndex} label={tag} size="small" />
                                    ))}
                                  </Box>
                                </Grid>
                                <Grid item xs={3}>
                                  <Box sx={{ textAlign: 'center' }}>
                                    <Typography variant="h4" color="primary">
                                      {Math.round((insight.confidence || 0.9) * 100)}%
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      Confidence
                                    </Typography>
                                  </Box>
                                </Grid>
                              </Grid>
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
                    ) : (
                      <Alert severity="warning">
                        No insights were generated. Check if there's data to analyze.
                      </Alert>
                    )}

                    {/* Show raw data for debugging */}
                    <details style={{ marginTop: '20px' }}>
                      <summary>Debug: View Raw Response Data</summary>
                      <Box
                        sx={{
                          p: 2,
                          bgcolor: 'background.default',
                          borderRadius: 1,
                          mt: 1,
                          maxHeight: 200,
                          overflow: 'auto',
                        }}
                      >
                        <Typography variant="body2" component="pre">
                          {JSON.stringify(analysisResult.data, null, 2)}
                        </Typography>
                      </Box>
                    </details>
                  </Box>
                ) : (
                  <Alert severity="error">
                    ❌ Failed to run analysis: {analysisResult.error}
                  </Alert>
                )}
              </Box>
            )}

            {!analysisResult && !analyzing && (
              <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 8 }}>
                Click "Run Analysis" to analyze your event data
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalyzerTab;