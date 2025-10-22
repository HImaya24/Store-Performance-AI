// components/dashboard/PricingTab.js
import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stepper,
  Step,
  StepLabel,
  TextField,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Alert,
  Container
} from '@mui/material';
import {
  Check as CheckIcon,
  Star as StarIcon,
  Business as BusinessIcon,
  Store as StoreIcon,
  Diamond as DiamondIcon,
  CreditCard as CreditCardIcon,
  Person as PersonIcon,
  Security as SecurityIcon
} from '@mui/icons-material';

const PricingTab = () => {
  const [openPayment, setOpenPayment] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState('');
  const [activeStep, setActiveStep] = useState(0);
  const [paymentSuccess, setPaymentSuccess] = useState(false);

  const pricingTiers = [
    {
      name: "Starter",
      price: "$0",
      period: "forever",
      description: "Perfect for small retail shops",
      icon: <StoreIcon sx={{ fontSize: 40 }} />,
      color: "#6366f1",
      popular: false,
      features: [
        "Up to 100 transactions/month",
        "Basic KPI dashboard",
        "Standard reports",
        "Email support",
        "1 store location",
        "30-day data retention"
      ],
      buttonText: "Get Started Free",
      buttonVariant: "outlined",
      action: (plan) => {
        alert(`🎉 ${plan.name} plan activated! You now have full access to Starter features.`);
      }
    },
    {
      name: "Professional",
      price: "$99",
      period: "per month",
      description: "For growing retail businesses", 
      icon: <StarIcon sx={{ fontSize: 40 }} />,
      color: "#8b5cf6",
      popular: true,
      features: [
        "Up to 10,000 transactions/month",
        "AI-powered insights",
        "Advanced analytics",
        "Semantic search",
        "Priority support", 
        "Custom dashboards",
        "5 store locations",
        "1-year data retention",
        "API access (basic)",
        "Seasonal trend analysis"
      ],
      buttonText: "Start 14-Day Free Trial",
      buttonVariant: "contained",
      action: (plan) => {
        setSelectedPlan(plan.name);
        setOpenPayment(true);
        setActiveStep(0);
        setPaymentSuccess(false);
      }
    },
    {
      name: "Enterprise", 
      price: "$499",
      period: "per month",
      description: "For large retail enterprises",
      icon: <DiamondIcon sx={{ fontSize: 40 }} />,
      color: "#10b981",
      popular: false,
      features: [
        "Unlimited transactions",
        "Real-time AI analysis",
        "Custom AI model training",
        "Full API access",
        "Custom integrations",
        "Dedicated account manager",
        "Unlimited store locations",
        "Unlimited data retention", 
        "SLA guarantee (99.9% uptime)",
        "White-label solutions",
        "On-premise deployment option",
        "Custom feature development"
      ],
      buttonText: "Contact Sales",
      buttonVariant: "outlined",
      action: (plan) => {
        window.open('mailto:sales@retail-ai.com?subject=Enterprise Plan Inquiry&body=Hello, I am interested in the Enterprise plan for my business.', '_blank');
      }
    }
  ];

  const businessMetrics = [
    {
      metric: "15-20%",
      label: "Average Revenue Increase",
      description: "From AI-powered cross-selling insights"
    },
    {
      metric: "80%",
      label: "Time Reduction", 
      description: "In manual data analysis tasks"
    },
    {
      metric: "200%+",
      label: "ROI",
      description: "Within first 3 months for most customers"
    },
    {
      metric: "24/7",
      label: "AI Monitoring",
      description: "Continuous store performance tracking"
    }
  ];

  const steps = ['Plan Selection', 'Account Info', 'Payment', 'Confirmation'];

  const handlePlanSelect = (plan) => {
    plan.action(plan);
  };

  const handleNext = () => {
    if (activeStep === steps.length - 2) {
      // Simulate payment processing
      setTimeout(() => {
        setPaymentSuccess(true);
        setActiveStep((prev) => prev + 1);
      }, 2000);
    } else {
      setActiveStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleClose = () => {
    setOpenPayment(false);
    setActiveStep(0);
    setPaymentSuccess(false);
  };

  const handleComplete = () => {
    handleClose();
    // Show success message
    setTimeout(() => {
      alert(`🚀 Welcome to ${selectedPlan}! Your 14-day free trial has started. Check your email for confirmation.`);
    }, 500);
  };

  const PaymentModal = () => (
    <Dialog open={openPayment} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ 
        textAlign: 'center', 
        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
        color: 'white',
        py: 3
      }}>
        <CreditCardIcon sx={{ mr: 1, fontSize: 32 }} />
        {selectedPlan} Plan - 14-Day Free Trial
      </DialogTitle>
      
      <DialogContent sx={{ p: 4 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {activeStep === 0 && (
          <Box>
            <Typography variant="h5" gutterBottom color="primary" textAlign="center">
              Plan Details
            </Typography>
            <Paper sx={{ p: 4, background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
              <Typography variant="h4" gutterBottom color="white" textAlign="center">
                {selectedPlan} Plan
              </Typography>
              <Typography variant="h6" color="text.secondary" textAlign="center" gutterBottom>
                14-Day Free Trial • Then ${selectedPlan === 'Professional' ? '99' : '499'}/month
              </Typography>
              
              <List sx={{ mt: 2 }}>
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon sx={{ color: '#10b981' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Full access to all features during trial" 
                    primaryTypographyProps={{ color: 'white' }}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon sx={{ color: '#10b981' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="No credit card required for trial" 
                    primaryTypographyProps={{ color: 'white' }}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon sx={{ color: '#10b981' }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Cancel anytime during trial period" 
                    primaryTypographyProps={{ color: 'white' }}
                  />
                </ListItem>
              </List>
            </Paper>
          </Box>
        )}

        {activeStep === 1 && (
          <Box>
            <Typography variant="h5" gutterBottom color="primary" textAlign="center">
              Account Information
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TextField 
                  fullWidth 
                  label="Full Name" 
                  variant="outlined" 
                  placeholder="John Doe"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField 
                  fullWidth 
                  label="Company Name" 
                  variant="outlined" 
                  placeholder="Retail Store Inc"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField 
                  fullWidth 
                  label="Email Address" 
                  type="email" 
                  variant="outlined"
                  placeholder="john@company.com"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField 
                  fullWidth 
                  label="Phone Number" 
                  variant="outlined"
                  placeholder="+1 (555) 123-4567"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField 
                  fullWidth 
                  label="Store Website" 
                  variant="outlined"
                  placeholder="https://yourstore.com"
                />
              </Grid>
            </Grid>
          </Box>
        )}

        {activeStep === 2 && (
          <Box>
            <Typography variant="h5" gutterBottom color="primary" textAlign="center">
              Payment Method
            </Typography>
            <FormControl component="fieldset" sx={{ width: '100%', mb: 3 }}>
              <FormLabel component="legend" sx={{ color: 'white', mb: 2, fontSize: '1.1rem' }}>
                Select Payment Method
              </FormLabel>
              <RadioGroup defaultValue="card">
                <FormControlLabel 
                  value="card" 
                  control={<Radio sx={{ color: 'white' }} />} 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', color: 'white' }}>
                      <CreditCardIcon sx={{ mr: 2 }} />
                      <Box>
                        <Typography variant="body1" fontWeight="bold">Credit/Debit Card</Typography>
                        <Typography variant="body2" color="text.secondary">Visa, Mastercard, American Express</Typography>
                      </Box>
                    </Box>
                  } 
                  sx={{ mb: 2, p: 2, border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}
                />
                <FormControlLabel 
                  value="paypal" 
                  control={<Radio sx={{ color: 'white' }} />} 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', color: 'white' }}>
                      <PersonIcon sx={{ mr: 2 }} />
                      <Box>
                        <Typography variant="body1" fontWeight="bold">PayPal</Typography>
                        <Typography variant="body2" color="text.secondary">Pay with your PayPal account</Typography>
                      </Box>
                    </Box>
                  } 
                  sx={{ mb: 2, p: 2, border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}
                />
                <FormControlLabel 
                  value="bank" 
                  control={<Radio sx={{ color: 'white' }} />} 
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', color: 'white' }}>
                      <SecurityIcon sx={{ mr: 2 }} />
                      <Box>
                        <Typography variant="body1" fontWeight="bold">Bank Transfer</Typography>
                        <Typography variant="body2" color="text.secondary">Wire transfer or ACH</Typography>
                      </Box>
                    </Box>
                  } 
                  sx={{ p: 2, border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}
                />
              </RadioGroup>
            </FormControl>
            
            <Alert severity="info" sx={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <Typography variant="body2" fontWeight="bold">
                💡 No payment required during 14-day free trial
              </Typography>
              <Typography variant="body2">
                You'll only be charged if you decide to continue after the trial period ends.
              </Typography>
            </Alert>
          </Box>
        )}

        {activeStep === 3 && paymentSuccess && (
          <Box textAlign="center">
            <CheckIcon sx={{ fontSize: 80, color: '#10b981', mb: 3 }} />
            <Typography variant="h4" gutterBottom color="primary" fontWeight="bold">
              🎉 Trial Activated Successfully!
            </Typography>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Welcome to the {selectedPlan} Plan
            </Typography>
            <Paper sx={{ p: 4, mt: 3, background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <Typography variant="h6" gutterBottom color="white">
                🚀 Getting Started Guide
              </Typography>
              <Box sx={{ textAlign: 'left' }}>
                <Typography variant="body2" color="white" gutterBottom>
                  1. 📥 <strong>Upload your data</strong> in the Collector tab
                </Typography>
                <Typography variant="body2" color="white" gutterBottom>
                  2. 📊 <strong>Generate KPIs</strong> to see AI insights
                </Typography>
                <Typography variant="body2" color="white" gutterBottom>
                  3. 🔍 <strong>Explore analytics</strong> in the Analyzer tab
                </Typography>
                <Typography variant="body2" color="white" gutterBottom>
                  4. 📈 <strong>View reports</strong> with AI-powered insights
                </Typography>
                <Typography variant="body2" color="white" gutterBottom>
                  5. 💡 <strong>Get recommendations</strong> for business growth
                </Typography>
              </Box>
            </Paper>
            
            <Alert severity="success" sx={{ mt: 3, background: 'rgba(16, 185, 129, 0.1)' }}>
              <Typography variant="body2">
                ✅ Confirmation email sent • ✅ Full access granted • ✅ Trial ends in 14 days
              </Typography>
            </Alert>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ justifyContent: 'space-between', p: 3, background: 'rgba(30, 41, 59, 0.8)' }}>
        <Button 
          onClick={activeStep === 0 ? handleClose : handleBack}
          disabled={activeStep === 3}
          sx={{ color: 'white' }}
        >
          {activeStep === 0 ? 'Cancel' : 'Back'}
        </Button>
        <Box>
          {activeStep < steps.length - 1 && (
            <Button 
              variant="contained" 
              onClick={handleNext}
              disabled={activeStep === 2 && !paymentSuccess}
              sx={{
                background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                px: 4,
                py: 1
              }}
            >
              {activeStep === steps.length - 2 ? 'Complete Signup' : 'Continue'}
            </Button>
          )}
          {activeStep === steps.length - 1 && (
            <Button 
              variant="contained" 
              onClick={handleComplete}
              sx={{ 
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                px: 4,
                py: 1
              }}
            >
              Start Using {selectedPlan}
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" fontWeight="bold" gutterBottom color="primary">
          💰 Pricing Plans
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Choose the perfect plan for your retail business. Start free, upgrade as you grow.
        </Typography>
      </Box>

      {/* Pricing Tiers */}
      <Grid container spacing={4} sx={{ mb: 8 }}>
        {pricingTiers.map((tier, index) => (
          <Grid item xs={12} md={4} key={tier.name}>
            <Card 
              sx={{ 
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                border: tier.popular ? `2px solid ${tier.color}` : '1px solid rgba(255,255,255,0.1)',
                background: tier.popular 
                  ? `linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)`
                  : 'rgba(30, 41, 59, 0.8)',
                backdropFilter: 'blur(10px)',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: `0 20px 40px rgba(0,0,0,0.3)`
                }
              }}
            >
              {tier.popular && (
                <Chip 
                  label="MOST POPULAR" 
                  color="primary"
                  sx={{ 
                    position: 'absolute', 
                    top: -12, 
                    left: '50%', 
                    transform: 'translateX(-50%)',
                    fontWeight: 'bold'
                  }}
                />
              )}
              
              <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Tier Header */}
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Box sx={{ color: tier.color, mb: 2 }}>
                    {tier.icon}
                  </Box>
                  <Typography variant="h4" fontWeight="bold" color="white" gutterBottom>
                    {tier.name}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {tier.description}
                  </Typography>
                </Box>

                {/* Price */}
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Typography variant="h2" fontWeight="bold" color="white" gutterBottom>
                    {tier.price}
                  </Typography>
                  <Typography variant="h6" color="text.secondary">
                    {tier.period}
                  </Typography>
                </Box>

                {/* Features */}
                <List sx={{ mb: 3, flexGrow: 1 }}>
                  {tier.features.map((feature, featureIndex) => (
                    <ListItem key={featureIndex} sx={{ px: 0, py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <CheckIcon sx={{ color: tier.color, fontSize: 20 }} />
                      </ListItemIcon>
                      <ListItemText 
                        primary={feature} 
                        primaryTypographyProps={{ 
                          variant: 'body2', 
                          color: 'white',
                          fontSize: '0.9rem'
                        }} 
                      />
                    </ListItem>
                  ))}
                </List>

                {/* Action Button */}
                <Button
                  variant={tier.buttonVariant}
                  fullWidth
                  size="large"
                  onClick={() => handlePlanSelect(tier)}
                  sx={{
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    background: tier.popular 
                      ? `linear-gradient(135deg, ${tier.color} 0%, #7c4cf0 100%)`
                      : 'transparent',
                    color: tier.popular ? 'white' : tier.color,
                    border: tier.popular ? 'none' : `2px solid ${tier.color}`,
                    '&:hover': {
                      background: tier.popular 
                        ? `linear-gradient(135deg, #5b5cdc 0%, #6a3dc8 100%)`
                        : `rgba(99, 102, 241, 0.1)`,
                      transform: 'translateY(-2px)'
                    },
                    transition: 'all 0.3s ease'
                  }}
                >
                  {tier.buttonText}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Business Impact Metrics */}
      <Paper sx={{ p: 4, mb: 6, background: 'rgba(30, 41, 59, 0.8)' }}>
        <Typography variant="h4" textAlign="center" gutterBottom color="primary">
          📈 Expected Business Impact
        </Typography>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          {businessMetrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <Typography variant="h3" color="#10b981" fontWeight="bold" gutterBottom>
                  {metric.metric}
                </Typography>
                <Typography variant="h6" color="white" gutterBottom>
                  {metric.label}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metric.description}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Business Model Section */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 4, background: 'rgba(30, 41, 59, 0.8)' }}>
            <Typography variant="h5" gutterBottom color="primary">
              🚀 Business Model
            </Typography>
            <List>
              <ListItem>
                <ListItemIcon>
                  <BusinessIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="SaaS Subscription Model"
                  secondary="Monthly recurring revenue with predictable growth"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <BusinessIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Value-Based Pricing"
                  secondary="Priced on business value delivered, not just features"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <BusinessIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="Tiered Approach"
                  secondary="Scales from individual stores to enterprise chains"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 4, background: 'rgba(30, 41, 59, 0.8)' }}>
            <Typography variant="h5" gutterBottom color="primary">
              🎯 Target Market
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="🏪 Small Retail Stores"
                  secondary="1-5 locations, basic analytics needs"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="🏬 Regional Retail Chains" 
                  secondary="5-50 locations, advanced insights"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="🏢 Large Retail Enterprises"
                  secondary="50+ locations, custom AI solutions"
                  primaryTypographyProps={{ color: 'white', fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'text.secondary' }}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Financial Projections */}
      <Paper sx={{ p: 4, mt: 4, background: 'rgba(30, 41, 59, 0.8)' }}>
        <Typography variant="h4" textAlign="center" gutterBottom color="primary">
          📊 Financial Projections
        </Typography>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card sx={{ textAlign: 'center', p: 3, background: 'rgba(99, 102, 241, 0.1)' }}>
              <Typography variant="h6" color="primary">Year 1</Typography>
              <Typography variant="h3" color="white" gutterBottom>$144K</Typography>
              <Typography variant="body2" color="text.secondary">Annual Revenue</Typography>
              <Typography variant="body2" color="text.secondary">100 Customers</Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ textAlign: 'center', p: 3, background: 'rgba(139, 92, 246, 0.1)' }}>
              <Typography variant="h6" color="primary">Year 2</Typography>
              <Typography variant="h3" color="white" gutterBottom>$500K</Typography>
              <Typography variant="body2" color="text.secondary">Annual Revenue</Typography>
              <Typography variant="body2" color="text.secondary">350 Customers</Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ textAlign: 'center', p: 3, background: 'rgba(16, 185, 129, 0.1)' }}>
              <Typography variant="h6" color="primary">Year 3</Typography>
              <Typography variant="h3" color="white" gutterBottom>$1M+</Typography>
              <Typography variant="body2" color="text.secondary">Annual Revenue</Typography>
              <Typography variant="body2" color="text.secondary">800 Customers</Typography>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* CTA Section */}
      <Box sx={{ textAlign: 'center', mt: 6, p: 4 }}>
        <Typography variant="h4" gutterBottom color="primary">
          Ready to Transform Your Retail Business?
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
          Join hundreds of retailers using AI to boost their revenue
        </Typography>
        <Button 
          variant="contained" 
          size="large"
          onClick={() => handlePlanSelect(pricingTiers[1])} // Professional plan
          sx={{
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            px: 4,
            py: 1.5,
            fontSize: '1.2rem',
            fontWeight: 'bold'
          }}
        >
          Start Your Free Trial Today
        </Button>
      </Box>

      {/* Payment Modal */}
      <PaymentModal />
    </Box>
  );
};

export default PricingTab;