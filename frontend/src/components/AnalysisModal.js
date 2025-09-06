import React from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, Typography, Box, Chip, Divider, Grid,
  List, ListItem, ListItemText
} from '@mui/material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

function AnalysisModal({ analysis, open, onClose }) {
  if (!analysis) return null;

  const getSeverityColor = (severity) => {
    const colors = {
      low: '#4caf50',
      medium: '#ff9800',
      high: '#f44336',
      critical: '#d32f2f'
    };
    return colors[severity] || colors.medium;
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">
            Error Analysis Details
          </Typography>
          <Chip
            label={analysis.severity_level}
            sx={{
              backgroundColor: getSeverityColor(analysis.severity_level),
              color: 'white'
            }}
          />
        </Box>
      </DialogTitle>
      
      <DialogContent dividers>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Failure Category</strong>
            </Typography>
            <Typography variant="body2" paragraph>
              {analysis.failure_category}
            </Typography>
            
            <Typography variant="subtitle1" gutterBottom>
              <strong>Summary</strong>
            </Typography>
            <Typography variant="body2" paragraph>
              {analysis.summary}
            </Typography>
            
            <Typography variant="subtitle1" gutterBottom>
              <strong>Environment & Server</strong>
            </Typography>
            <Box display="flex" gap={1} mb={2}>
              <Chip label={analysis.environment} variant="outlined" />
              <Chip label={analysis.server} variant="outlined" />
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Metrics</strong>
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Business Impact"
                  secondary={`${(analysis.business_impact_score * 100).toFixed(0)}%`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Confidence Score"
                  secondary={`${(analysis.confidence_score * 100).toFixed(0)}%`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Error Count"
                  secondary={analysis.error_count}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Resolution Time"
                  secondary={analysis.resolution_time_estimate}
                />
              </ListItem>
            </List>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        {analysis.root_cause && analysis.root_cause.cause && (
          <Box mb={3}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Root Cause Analysis</strong>
            </Typography>
            <Typography variant="body2" paragraph>
              <strong>Cause:</strong> {analysis.root_cause.cause}
            </Typography>
            {analysis.root_cause.reasoning_chain && (
              <Typography variant="body2" paragraph>
                <strong>Reasoning:</strong> {analysis.root_cause.reasoning_chain}
              </Typography>
            )}
          </Box>
        )}
        
        {analysis.fix_suggestion && analysis.fix_suggestion.command && (
          <Box mb={3}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Fix Suggestion</strong>
            </Typography>
            <SyntaxHighlighter language="bash" style={tomorrow}>
              {analysis.fix_suggestion.command}
            </SyntaxHighlighter>
            {analysis.fix_suggestion.steps && (
              <Box mt={2}>
                <Typography variant="body2" gutterBottom>
                  <strong>Steps:</strong>
                </Typography>
                <List dense>
                  {analysis.fix_suggestion.steps.map((step, index) => (
                    <ListItem key={index}>
                      <ListItemText primary={`${index + 1}. ${step}`} />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Box>
        )}
        
        {analysis.affected_components && analysis.affected_components.length > 0 && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Affected Components</strong>
            </Typography>
            <Box display="flex" gap={1} flexWrap="wrap">
              {analysis.affected_components.map((component, index) => (
                <Chip key={index} label={component} size="small" />
              ))}
            </Box>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default AnalysisModal;
