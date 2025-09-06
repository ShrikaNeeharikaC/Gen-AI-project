import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Grid, Card, CardContent, Typography, Box, CircularProgress,
  Alert, Select, MenuItem, FormControl, InputLabel, Tabs, Tab,
  List, ListItem, ListItemText, Paper, Accordion, AccordionSummary,
  AccordionDetails, Chip, Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import axios from 'axios';
import MetricsCard from './MetricsCard';
import AnalysisCard from './AnalysisCard';
import AnalysisModal from './AnalysisModal';

function ProjectDetails() {
  const { toolName, projectName } = useParams();
  const [metrics, setMetrics] = useState(null);
  const [analyses, setAnalyses] = useState([]);
  const [environments, setEnvironments] = useState([]);
  const [servers, setServers] = useState([]);
  const [pipelineStages, setPipelineStages] = useState({
    "git-checkout": [],
    "build": [],
    "test": [],
    "sonarqube-issues": []
  });
  const [selectedEnv, setSelectedEnv] = useState('');
  const [selectedServer, setSelectedServer] = useState('');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState({
    metrics: true,
    analyses: true,
    environments: true,
    servers: false,
    logs: false
  });

  useEffect(() => {
    fetchProjectData();
  }, [toolName, projectName]);

  useEffect(() => {
    if (selectedEnv) {
      fetchServers();
      fetchPipelineStages();
    } else {
      setServers([]);
      setSelectedServer('');
    }
  }, [selectedEnv]);

  useEffect(() => {
    if (selectedEnv && selectedServer) {
      fetchPipelineStages();
    }
  }, [selectedEnv, selectedServer]);

  const fetchProjectData = async () => {
    try {
      // Fetch metrics
      setLoading(prev => ({ ...prev, metrics: true }));
      const metricsResponse = await axios.get(`/project-metrics/${toolName}/${projectName}`);
      setMetrics(metricsResponse.data);
      setLoading(prev => ({ ...prev, metrics: false }));

      // Fetch analyses
      setLoading(prev => ({ ...prev, analyses: true }));
      const analysesResponse = await axios.get(`/project-metrics/${toolName}/${projectName}`);
      setAnalyses(analysesResponse.data);
      setLoading(prev => ({ ...prev, analyses: false }));

      // Ensure response is an array
      const analysesData = Array.isArray(analysesResponse.data) 
        ? analysesResponse.data 
        : [];
      setAnalyses(analysesData);
      setLoading(prev => ({ ...prev, analyses: false }));


      // Fetch environments
      setLoading(prev => ({ ...prev, environments: true }));
      const envsResponse = await axios.get(`/environments/${toolName}/${projectName}`);
      setEnvironments(envsResponse.data);
      setLoading(prev => ({ ...prev, environments: false }));

      fetchPipelineStages();

    } catch (error) {
      console.error('Error fetching project data:', error);
      setLoading({
        metrics: false,
        analyses: false,
        environments: false,
        servers: false,
        logs: false
      });
    }
  };

  const fetchServers = async () => {
    try {
      setLoading(prev => ({ ...prev, servers: true }));
      const response = await axios.get(`/servers/${toolName}/${projectName}/${selectedEnv}`);
      setServers(response.data);
    } catch (error) {
      console.error('Error fetching servers:', error);
    } finally {
      setLoading(prev => ({ ...prev, servers: false }));
    }
  };


  const fetchPipelineStages = async () => {
    try{
      setLoading(prev => ({ ...prev, stages: true }));
      let url = `/pipeline-stages/${toolName}/${projectName}`;
      if (selectedEnv) {
        url += `/${selectedEnv}`;
        if (selectedServer) {
          url += `/${selectedServer}`;
        }
      }

      const response = await axios.get(url);
      setPipelineStages(response.data);
    }
    catch (error) {
      console.error('Error fetching pipeline stages:', error);
    }
    finally {
      setLoading(prev => ({ ...prev, stages: false }));
    }
  }

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getSeverityColor = (severity) => {
    const colors = {
      low: '#4caf50',
      medium: '#ff9800',
      high: '#f44336',
      critical: '#d32f2f'
    };
    return colors[severity] || colors.medium;
  };

  const renderStageAnalysis = (analysis) => (
    <Accordion key={analysis.id}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Box display="flex" alignItems="center" width="100%">
          <Typography sx={{ flexGrow: 1 }}>
            {new Date(analysis.timestamp).toLocaleString()}
          </Typography>
          <Chip
            label={analysis.severity_level}
            size="small"
            sx={{
              backgroundColor: getSeverityColor(analysis.severity_level),
              color: 'white',
              mr: 1
            }}
          />
          <Chip
            label={`${(analysis.confidence_score * 100).toFixed(0)}%`}
            size="small"
            variant="outlined"
          />
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            <strong>Failure Summary</strong>
          </Typography>
          <Typography variant="body2" paragraph>
            {analysis.analysis.failure_summary}
          </Typography>

          {analysis.analysis.root_cause?.cause && (
            <Box mb={2}>
              <Typography variant="subtitle2" gutterBottom>
                <strong>Root Cause</strong>
              </Typography>
              <Typography variant="body2" paragraph>
                {analysis.analysis.root_cause.cause}
              </Typography>
              {analysis.analysis.root_cause.reasoning_chain && (
                <Typography variant="body2" color="textSecondary">
                  {analysis.analysis.root_cause.reasoning_chain}
                </Typography>
              )}
            </Box>
          )}

          {analysis.analysis.fix_suggestion?.command && (
            <Box mb={2}>
              <Typography variant="subtitle2" gutterBottom>
                <strong>Fix Suggestion</strong>
              </Typography>
              <SyntaxHighlighter language="bash" style={tomorrow}>
                {analysis.analysis.fix_suggestion.command}
              </SyntaxHighlighter>
              {analysis.analysis.fix_suggestion.steps && (
                <List dense>
                  {analysis.analysis.fix_suggestion.steps.map((step, index) => (
                    <ListItem key={index}>
                      <ListItemText primary={`${index + 1}. ${step}`} />
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          )}

          {analysis.analysis.rollback_plan?.commands && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                <strong>Rollback Plan</strong>
              </Typography>
              <SyntaxHighlighter language="bash" style={tomorrow}>
                {analysis.analysis.rollback_plan.commands.join('\n')}
              </SyntaxHighlighter>
            </Box>
          )}
        </Box>
      </AccordionDetails>
    </Accordion>
  );

  const tabLabels = ['Git-checkout', 'Build', 'Test', 'Sonarqube Issues'];
  const tabKeys = ['git-checkout', 'build', 'test', 'sonarqube-issues'];

  const getHealthColor = (score) => {
    if (score >= 80) return '#4caf50';
    if (score >= 60) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        {toolName} - {projectName}
      </Typography>

      {/* Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Success Rate"
            value={`${metrics?.success_rate || 0}%`}
            color={getHealthColor(metrics?.success_rate || 0)}
            loading={loading.metrics}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="MTTR"
            value={`${metrics?.mttr_hours || 0}h`}
            color="#2196f3"
            loading={loading.metrics}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Avg Build Time"
            value={`${metrics?.avg_build_time_minutes || 0}m`}
            color="#9c27b0"
            loading={loading.metrics}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Health Score"
            value={`${metrics?.health_score || 0}`}
            color={getHealthColor(metrics?.health_score || 0)}
            loading={loading.metrics}
          />
        </Grid>
      </Grid>

      {/* Charts and Environment Filter */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Environment Filter
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Environment</InputLabel>
                <Select
                  value={selectedEnv}
                  label="Environment"
                  onChange={(e) => setSelectedEnv(e.target.value)}
                >
                  <MenuItem value="">All Environments</MenuItem>
                  {environments.map((env) => (
                    <MenuItem key={env.name} value={env.name}>
                      {env.name} ({env.count})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth>
                <InputLabel>Server</InputLabel>
                <Select
                  value={selectedServer}
                  label="Server"
                  onChange={(e) => setSelectedServer(e.target.value)}
                  disabled={!selectedEnv || loading.servers}
                >
                  <MenuItem value="">All Servers</MenuItem>
                  {servers.map((server) => (
                    <MenuItem key={server.name} value={server.name}>
                      {server.name} (Health: {server.health_score}%)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Environment Filter
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Environment</InputLabel>
                <Select
                  value={selectedEnv}
                  label="Environment"
                  onChange={(e) => setSelectedEnv(e.target.value)}
                >
                  {environments.map((env) => (
                    <MenuItem key={env.name} value={env.name}>
                      {env.name} ({env.count})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth>
                <InputLabel>Server</InputLabel>
                <Select
                  value={selectedServer}
                  label="Server"
                  onChange={(e) => setSelectedServer(e.target.value)}
                  disabled={!selectedEnv || loading.servers}
                >
                  {servers.map((server) => (
                    <MenuItem key={server.name} value={server.name}>
                      <Box display="flex" alignItems="center" width="100%">
                        <Typography sx={{ flexGrow: 1 }}>
                          {server.name}
                        </Typography>
                        <Chip
                          size="small"
                          label={server.status}
                          sx={{
                            backgroundColor: getHealthColor(server.health_score),
                            color: 'white'
                          }}
                        />
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Error Analysis Cards */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Real-time Error Analysis
          </Typography>
          {loading.analyses ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : (
            <Grid container spacing={2}>
              {Array.isArray(analyses) && analyses.slice(0, 6).map((analysis) => (
                <Grid item xs={12} sm={6} md={4} key={analysis.id}>
                  <AnalysisCard
                    analysis={analysis}
                    onClick={() => setSelectedAnalysis(analysis)}
                  />
                </Grid>
              ))}
            </Grid>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Pipeline Stage Analysis
          </Typography>
          
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="pipeline stages">
            {tabLabels.map((label, index) => (
              <Tab 
                key={index} 
                label={`${label} (${pipelineStages[tabKeys[index]]?.length || 0})`} 
              />
            ))}
          </Tabs>

          <Box sx={{ mt: 2 }}>
            {loading.stages ? (
              <Box display="flex" justifyContent="center" p={4}>
                <CircularProgress />
              </Box>
            ) : (
              <Box>
                {pipelineStages[tabKeys[activeTab]]?.length > 0 ? (
                  pipelineStages[tabKeys[activeTab]].map(renderStageAnalysis)
                ) : (
                  <Typography variant="body2" color="textSecondary" sx={{ p: 2 }}>
                    No analysis available for {tabLabels[activeTab].toLowerCase()}
                  </Typography>
                )}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Analysis Detail Modal */}
      <AnalysisModal
        analysis={selectedAnalysis}
        open={!!selectedAnalysis}
        onClose={() => setSelectedAnalysis(null)}
      />
    </Box>
  );
}

export default ProjectDetails;
