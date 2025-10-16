import React from 'react';
import {
  Box,
  Typography,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  alpha,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Store as StoreIcon,
  Report as ReportIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';

const LandingPage = ({ onGetStarted }) => {
  const features = [
    {
      icon: <AnalyticsIcon sx={{ fontSize: 40 }} />,
      title: 'Real-time Analytics',
      description: 'Live store performance monitoring with instant insights.',
      color: '#6366f1',
    },
    {
      icon: <StoreIcon sx={{ fontSize: 40 }} />,
      title: 'Multi-Store Dashboard',
      description: 'Unified view of all your stores with comparative analytics.',
      color: '#ec4899',
    },
    {
      icon: <ReportIcon sx={{ fontSize: 40 }} />,
      title: 'Smart Reporting',
      description: 'Automated reports with AI-powered insights.',
      color: '#10b981',
    },
    {
      icon: <TrendingUpIcon sx={{ fontSize: 40 }} />,
      title: 'Sales Intelligence',
      description: 'Predict trends and identify growth opportunities.',
      color: '#f59e0b',
    },
  ];

  return (
    <Box sx={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}>
      {/* Hero Section */}
      <Box
        sx={{
          minHeight: '100vh',
          background: `linear-gradient(
            rgba(15, 23, 42, 0.85), 
            rgba(30, 41, 59, 0.90)
          ), url('https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <Box sx={{ color: 'white' }}>
                {/* Badge */}
                <Box
                  sx={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    background: alpha('#6366f1', 0.1),
                    border: `1px solid ${alpha('#6366f1', 0.3)}`,
                    borderRadius: '20px',
                    px: 2,
                    py: 1,
                    mb: 3,
                  }}
                >
                  <Typography variant="body2" fontWeight="600" color="#6366f1">
                    🚀 AI-Powered Analytics Platform
                  </Typography>
                </Box>

                {/* Main Heading */}
                <Typography
                  variant="h2"
                  fontWeight="bold"
                  sx={{
                    mb: 3,
                    fontSize: { xs: '2.5rem', md: '3.5rem' },
                    lineHeight: 1.2,
                    background: 'linear-gradient(135deg, #fff 0%, #94a3b8 100%)',
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    color: 'transparent',
                  }}
                >
                  Store Performance AI Dashboard
                </Typography>

                {/* Subtitle */}
                <Typography
                  variant="h6"
                  sx={{
                    color: alpha('#fff', 0.8),
                    mb: 4,
                    lineHeight: 1.6,
                    fontSize: '1.25rem',
                  }}
                >
                  Monitor your retail AI agents and analyze transaction data in real-time. 
                  Make data-driven decisions that drive growth and optimize store performance.
                </Typography>

                {/* CTA Button - This now calls onGetStarted */}
                <Button
                  variant="contained"
                  size="large"
                  onClick={onGetStarted} // This triggers the navigation
                  sx={{
                    background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: '600',
                    borderRadius: '12px',
                    boxShadow: '0 10px 30px rgba(99, 102, 241, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #5b5cdc 0%, #7c4cf0 100%)',
                      boxShadow: '0 15px 40px rgba(99, 102, 241, 0.4)',
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  🚀 Launch Dashboard
                </Button>

                {/* Trust Indicators */}
                <Box sx={{ mt: 4, display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ display: 'flex' }}>
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Box key={star} sx={{ color: '#fbbf24', fontSize: '1.1rem' }}>
                          ⭐
                        </Box>
                      ))}
                    </Box>
                    <Typography variant="body2" color={alpha('#fff', 0.7)}>
                      4.9/5 (500+ stores)
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              {/* Features Grid */}
              <Grid container spacing={2}>
                {features.map((feature, index) => (
                  <Grid item xs={6} key={index}>
                    <Card
                      sx={{
                        background: alpha('#1e293b', 0.6),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${alpha('#fff', 0.1)}`,
                        borderRadius: '12px',
                        p: 2,
                        height: '100%',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          borderColor: alpha(feature.color, 0.5),
                        },
                      }}
                    >
                      <CardContent sx={{ textAlign: 'center', p: '8px !important' }}>
                        <Box
                          sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 60,
                            height: 60,
                            borderRadius: '12px',
                            background: `linear-gradient(135deg, ${alpha(feature.color, 0.2)} 0%, ${alpha(feature.color, 0.1)} 100%)`,
                            border: `1px solid ${alpha(feature.color, 0.3)}`,
                            mb: 2,
                          }}
                        >
                          <Box sx={{ color: feature.color }}>
                            {feature.icon}
                          </Box>
                        </Box>
                        <Typography variant="subtitle2" fontWeight="600" color="#fff" gutterBottom>
                          {feature.title}
                        </Typography>
                        <Typography variant="body2" color={alpha('#fff', 0.7)} sx={{ fontSize: '0.75rem' }}>
                          {feature.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
};

export default LandingPage;