import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, Typography, List, ListItem, 
  ListItemText, Chip, Box, CircularProgress, Alert
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Dashboard() {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/projects');
      setTools(response.data);
    } catch (err) {
      setError('Failed to fetch projects');
      console.error('Error fetching projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleProjectClick = (projectName, toolName) => {
    navigate(`/project/${toolName}/${projectName}`);
  };

  const getToolColor = (tool) => {
    const colors = {
      'github_actions': '#2188ff',
      'jenkins': '#d33833',
      'gitlab': '#fc6d26',
      'azure_devops': '#0078d4',
      'default': '#757575'
    };
    return colors[tool] || colors.default;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
        Projects Overview
      </Typography>
      
      <Grid container spacing={3}>
        {tools.map((tool) => (
          <Grid item xs={12} md={6} lg={4} key={tool.tool}>
            <Card 
              elevation={2}
              sx={{ 
                height: '100%',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                }
              }}
            >
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Typography variant="h6" component="h2">
                    {tool.tool.replace('_', ' ').toUpperCase()}
                  </Typography>
                  <Chip 
                    label={`${tool.total_builds} builds`}
                    size="small"
                    sx={{ 
                      backgroundColor: getToolColor(tool.tool),
                      color: 'white'
                    }}
                  />
                </Box>
                
                <List dense>
                  {tool.projects.map((project) => (
                    <ListItem
                      key={project.name}
                      button
                      onClick={() => handleProjectClick(project.name, tool.tool)}
                      sx={{
                        borderRadius: 1,
                        mb: 0.5,
                        '&:hover': {
                          backgroundColor: 'action.hover',
                        }
                      }}
                    >
                      <ListItemText
                        primary={project.name}
                        secondary={`${project.doc_count} analyses`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {tools.length === 0 && (
        <Box textAlign="center" py={8}>
          <Typography variant="h6" color="textSecondary">
            No projects found
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Make sure your CI/CD pipelines are sending data to Elasticsearch
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default Dashboard;
