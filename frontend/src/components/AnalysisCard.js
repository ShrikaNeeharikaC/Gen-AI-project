import React from 'react';
import { Card, CardContent, Typography, Chip, Box } from '@mui/material';
import moment from 'moment';

function AnalysisCard({ analysis, onClick }) {
  const getSeverityColor = (severity) => {
    const colors = {
      low: '#4caf50',
      medium: '#ff9800',
      high: '#f44336',
      critical: '#d32f2f'
    };
    return colors[severity] || colors.medium;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    return '#f44336';
  };

  return (
    <Card 
      elevation={1}
      sx={{ 
        cursor: 'pointer',
        transition: 'all 0.2s',
        '&:hover': {
          elevation: 3,
          transform: 'translateY(-2px)'
        }
      }}
      onClick={onClick}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
          <Chip
            label={analysis.severity_level}
            size="small"
            sx={{
              backgroundColor: getSeverityColor(analysis.severity_level),
              color: 'white',
              fontWeight: 'bold'
            }}
          />
          <Typography variant="caption" color="textSecondary">
            {moment(analysis.timestamp).fromNow()}
          </Typography>
        </Box>
        
        <Typography variant="subtitle1" gutterBottom>
          {analysis.failure_category}
        </Typography>
        
        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
          {analysis.summary.substring(0, 100)}...
        </Typography>
        
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" gap={1}>
            <Chip
              label={analysis.environment}
              size="small"
              variant="outlined"
            />
            <Chip
              label={analysis.server}
              size="small"
              variant="outlined"
            />
          </Box>
          
          <Box display="flex" alignItems="center">
            <Typography variant="caption" sx={{ mr: 1 }}>
              Confidence:
            </Typography>
            <Chip
              label={`${(analysis.confidence_score * 100).toFixed(0)}%`}
              size="small"
              sx={{
                backgroundColor: getConfidenceColor(analysis.confidence_score),
                color: 'white'
              }}
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

export default AnalysisCard;
