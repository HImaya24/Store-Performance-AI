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
  Slider,
  CircularProgress,
  Grid,
} from '@mui/material';
import { PlayArrow as PlayIcon } from '@mui/icons-material';
import { triggerDataProcessing } from '../../services/api';

const CoordinatorTab = () => {
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [processType, setProcessType] = useState('full_pipeline');
  const [eventCount, setEventCount] = useState(10);

  const handleProcessData = async () => {
    setProcessing(true);
    setResult(null);
    
    try {
      // First get data from collector
      const collectorResponse = await fetch('http://localhost:8100/events');
      if (!collectorResponse.ok) {
        throw new Error('Could not get data from collector');
      }
      
      let events = await collectorResponse.json();
      
      // Limit events to coordinator requirement 
      if (events.length > 20) {
        events = events.slice(0, 20);
      }
      
      // Send to coordinator with proper authentication
      const response = await fetch('http://localhost:8110/orchestrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': 'demo-key', 
        },
        body: JSON.stringify({ events }),
      });
      
      if (response.ok) {
        const resultData = await response.json();
        setResult({ success: true, data: resultData });
      } else {
        const errorText = await response.text();
        setResult({ success: false, error: `Coordinator returned status ${response.status}: ${errorText}` });
      }
    } catch (error) {
      setResult({ success: false, error: error.message });
    } finally {
      setProcessing(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        🔄 Coordinator Agent
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Process data through the coordinator pipeline. Coordinator limit: 20 events maximum.
        <br />
        <strong>API Key:</strong> demo-key
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Processing Configuration
            </Typography>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Process Type</InputLabel>
              <Select
                value={processType}
                label="Process Type"
                onChange={(e) => setProcessType(e.target.value)}
              >
                <MenuItem value="full_pipeline">Full Pipeline</MenuItem>
                <MenuItem value="quick_scan">Quick Scan</MenuItem>
              </Select>
            </FormControl>

            

            <Button
              variant="contained"
              startIcon={processing ? <CircularProgress size={20} /> : <PlayIcon />}
              onClick={handleProcessData}
              disabled={processing}
              fullWidth
              size="large"
            >
              {processing ? 'Processing...' : 'Process Data'}
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, minHeight: 200 }}>
            <Typography variant="h6" gutterBottom>
              Processing Results
            </Typography>

            {result && (
              <Box>
                {result.success ? (
                  <Box>
                    {result.data?.status !== 'rejected' ? (
                      <Alert severity="success" sx={{ mb: 2 }}>
                        ✅ Processing complete! Batch ID: {result.data?.batch_id}, Status: {result.data?.status}
                      </Alert>
                    ) : (
                      <Alert severity="error" sx={{ mb: 2 }}>
                        ❌ Processing rejected: {result.data?.message}
                      </Alert>
                    )}
                  </Box>
                ) : (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    ❌ Processing failed: {result.error}
                  </Alert>
                )}

                {result.data && (
                  <Box
                    sx={{
                      p: 2,
                      bgcolor: 'background.default',
                      borderRadius: 1,
                      mt: 2,
                      maxHeight: 200,
                      overflow: 'auto',
                    }}
                  >
                    <Typography variant="body2" component="pre">
                      {JSON.stringify(result.data, null, 2)}
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {!result && !processing && (
              <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                Click "Process Data" to start processing
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CoordinatorTab;