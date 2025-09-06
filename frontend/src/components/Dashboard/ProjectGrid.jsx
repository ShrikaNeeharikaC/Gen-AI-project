// src/components/Dashboard/ProjectGrid.jsx - Professional layout with real data
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, AlertTriangle, CheckCircle, Clock, Server, TrendingUp, 
  TrendingDown, RefreshCw, Search, Filter, BarChart3, AlertCircle,
  GitBranch, Zap, Shield, Bug
} from 'lucide-react';

const ProjectGrid = ({ projects, loading, onRefresh }) => {
  const navigate = useNavigate();
  const [groupedProjects, setGroupedProjects] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTool, setFilterTool] = useState('all');
  const [sortBy, setSortBy] = useState('successRate');

  useEffect(() => {
    // Group and filter projects
    let filteredProjects = projects;
    
    // Apply search filter
    if (searchQuery) {
      filteredProjects = projects.filter(project => 
        project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.tool.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    // Apply tool filter
    if (filterTool !== 'all') {
      filteredProjects = filteredProjects.filter(project => project.tool === filterTool);
    }
    
    // Sort projects
    filteredProjects.sort((a, b) => {
      switch (sortBy) {
        case 'successRate':
          return b.successRate - a.successRate;
        case 'name':
          return a.name.localeCompare(b.name);
        case 'lastBuild':
          return new Date(b.lastBuild) - new Date(a.lastBuild);
        default:
          return 0;
      }
    });
    
    // Group by tool
    const grouped = filteredProjects.reduce((acc, project) => {
      const tool = project.tool;
      if (!acc[tool]) {
        acc[tool] = [];
      }
      acc[tool].push(project);
      return acc;
    }, {});
    
    setGroupedProjects(grouped);
  }, [projects, searchQuery, filterTool, sortBy]);

  const getToolIcon = (tool) => {
    const icons = {
      jenkins: '🔧',
      github_actions: '🐙',
      gitlab_ci: '🦊',
      azure_devops: '🔷',
      circleci: '⭕',
      bitbucket: '🪣',
      travis_ci: '🔨'
    };
    return icons[tool] || '🔨';
  };

  const getToolDisplayName = (tool) => {
    const names = {
      jenkins: 'Jenkins',
      github_actions: 'GitHub Actions',
      gitlab_ci: 'GitLab CI',
      azure_devops: 'Azure DevOps',
      circleci: 'CircleCI',
      bitbucket: 'Bitbucket Pipelines',
      travis_ci: 'Travis CI'
    };
    return names[tool] || tool.replace('_', ' ').toUpperCase();
  };

  const getStatusColor = (successRate) => {
    if (successRate >= 90) return 'status-success';
    if (successRate >= 70) return 'status-warning';
    return 'status-error';
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'severity-critical';
      case 'high': return 'severity-high';
      case 'medium': return 'severity-medium';
      case 'low': return 'severity-low';
      default: return 'severity-unknown';
    }
  };

  const getToolStats = (toolProjects) => {
    const totalProjects = toolProjects.length;
    const avgSuccessRate = Math.round(
      toolProjects.reduce((sum, p) => sum + p.successRate, 0) / totalProjects
    );
    const criticalIssues = toolProjects.reduce((sum, p) => sum + (p.criticalIssues || 0), 0);
    const totalAnalyses = toolProjects.reduce((sum, p) => sum + (p.totalAnalyses || 0), 0);
    
    return { totalProjects, avgSuccessRate, criticalIssues, totalAnalyses };
  };

  const uniqueTools = [...new Set(projects.map(p => p.tool))];

  if (loading) {
    return (
      <div className="project-grid-loading">
        <div className="loading-content-professional">
          <div className="loading-spinner-large"></div>
          <h2>Loading CI/CD Projects</h2>
          <p>Fetching real-time data from Elasticsearch...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="project-grid-container-professional">
      {/* Enhanced Page Header */}
      <div className="page-header-professional">
        <div className="header-content">
          <div className="title-section">
            <h1 className="page-title-professional">
              <Activity className="title-icon" size={32} />
              CI/CD Pipeline Dashboard
            </h1>
            <p className="page-subtitle">
              Real-time analytics and monitoring for your CI/CD pipelines
            </p>
            <div className="stats-summary">
              <div className="stat-item">
                <span className="stat-value">{projects.length}</span>
                <span className="stat-label">Total Projects</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{uniqueTools.length}</span>
                <span className="stat-label">CI/CD Tools</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {projects.reduce((sum, p) => sum + (p.totalAnalyses || 0), 0)}
                </span>
                <span className="stat-label">Total Analyses</span>
              </div>
            </div>
          </div>
          
          <div className="header-actions">
            <button className="refresh-btn-professional" onClick={onRefresh}>
              <RefreshCw size={16} />
              Refresh Data
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Controls */}
      <div className="controls-section-professional">
        <div className="search-and-filter">
          <div className="search-box-professional">
            <Search size={16} className="search-icon" />
            <input
              type="text"
              placeholder="Search projects or tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>
          
          <div className="filter-controls">
            <select 
              value={filterTool} 
              onChange={(e) => setFilterTool(e.target.value)}
              className="tool-filter"
            >
              <option value="all">All Tools</option>
              {uniqueTools.map(tool => (
                <option key={tool} value={tool}>
                  {getToolDisplayName(tool)}
                </option>
              ))}
            </select>
            
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value)}
              className="sort-select"
            >
              <option value="successRate">Sort by Success Rate</option>
              <option value="name">Sort by Name</option>
              <option value="lastBuild">Sort by Last Build</option>
            </select>
          </div>
        </div>
      </div>

      {/* Projects Display */}
      {Object.keys(groupedProjects).length === 0 ? (
        <div className="no-projects-professional">
          <BarChart3 size={64} className="no-projects-icon" />
          <h3>No Projects Found</h3>
          <p>No projects match your current filters or no data available in the database.</p>
          <button onClick={onRefresh} className="retry-btn-professional">
            <RefreshCw size={16} />
            Refresh Data
          </button>
        </div>
      ) : (
        Object.entries(groupedProjects).map(([tool, toolProjects]) => {
          const toolStats = getToolStats(toolProjects);
          
          return (
            <div key={tool} className="tool-section-professional">
              {/* Enhanced Tool Header */}
              <div className="tool-header-professional">
                <div className="tool-info">
                  <div className="tool-title-section">
                    <span className="tool-icon-large">{getToolIcon(tool)}</span>
                    <div className="tool-details">
                      <h2 className="tool-title">{getToolDisplayName(tool)}</h2>
                      <div className="tool-stats">
                        <span className="tool-stat">
                          {toolStats.totalProjects} projects
                        </span>
                        <span className="tool-stat">
                          {toolStats.avgSuccessRate}% avg success
                        </span>
                        <span className="tool-stat">
                          {toolStats.totalAnalyses} analyses
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="tool-health">
                  <div className={`health-indicator ${getStatusColor(toolStats.avgSuccessRate)}`}>
                    {toolStats.avgSuccessRate >= 90 ? (
                      <CheckCircle size={20} />
                    ) : toolStats.avgSuccessRate >= 70 ? (
                      <AlertTriangle size={20} />
                    ) : (
                      <AlertCircle size={20} />
                    )}
                    <span>
                      {toolStats.avgSuccessRate >= 90 ? 'Healthy' : 
                       toolStats.avgSuccessRate >= 70 ? 'Warning' : 'Critical'}
                    </span>
                  </div>
                  
                  {toolStats.criticalIssues > 0 && (
                    <div className="critical-issues-badge">
                      <Bug size={16} />
                      <span>{toolStats.criticalIssues} critical</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Enhanced Projects Grid */}
              <div className="projects-grid-professional">
                {toolProjects.map((project) => (
                  <div
                    key={`${tool}-${project.name}`}
                    className={`project-card-professional ${getStatusColor(project.successRate)}`}
                    onClick={() => navigate(`/project/${tool}/${project.name}`)}
                  >
                    {/* Project Header */}
                    <div className="project-header-professional">
                      <div className="project-title-section">
                        <h3 className="project-name">{project.name}</h3>
                        <div className={`severity-badge ${getSeverityColor(project.latestSeverity)}`}>
                          {project.latestSeverity}
                        </div>
                      </div>
                      
                      <div className="project-status-indicator">
                        {project.successRate >= 90 ? (
                          <CheckCircle size={20} className="status-icon success" />
                        ) : project.successRate >= 70 ? (
                          <AlertTriangle size={20} className="status-icon warning" />
                        ) : (
                          <AlertCircle size={20} className="status-icon error" />
                        )}
                      </div>
                    </div>

                    {/* Project Metrics */}
                    <div className="project-metrics-professional">
                      <div className="metric-row">
                        <div className="metric">
                          <div className="metric-icon">
                            <TrendingUp size={16} />
                          </div>
                          <div className="metric-content">
                            <span className="metric-value">{project.successRate}%</span>
                            <span className="metric-label">Success Rate</span>
                          </div>
                        </div>
                        
                        <div className="metric">
                          <div className="metric-icon">
                            <Clock size={16} />
                          </div>
                          <div className="metric-content">
                            <span className="metric-value">{project.lastBuild}</span>
                            <span className="metric-label">Last Build</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="metric-row">
                        <div className="metric">
                          <div className="metric-icon">
                            <BarChart3 size={16} />
                          </div>
                          <div className="metric-content">
                            <span className="metric-value">{project.totalAnalyses || 0}</span>
                            <span className="metric-label">Analyses</span>
                          </div>
                        </div>
                        
                        <div className="metric">
                          <div className="metric-icon">
                            <Shield size={16} />
                          </div>
                          <div className="metric-content">
                            <span className="metric-value">{project.avgConfidence || 0}%</span>
                            <span className="metric-label">Confidence</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Environment Status */}
                    <div className="project-environments-professional">
                      <div className="environments-label">Environments:</div>
                      <div className="environment-indicators">
                        {Object.entries(project.envStatus || {}).map(([env, status]) => (
                          <div
                            key={env}
                            className={`env-indicator ${status}`}
                            title={`${env}: ${status}`}
                          >
                            {env.charAt(0).toUpperCase()}
                          </div>
                        ))}
                        {Object.keys(project.envStatus || {}).length === 0 && (
                          <span className="no-env-data">No environment data</span>
                        )}
                      </div>
                    </div>

                    {/* Error Indicators */}
                    {(project.errorAnalyses > 0 || project.criticalIssues > 0) && (
                      <div className="project-alerts">
                        {project.criticalIssues > 0 && (
                          <div className="alert-item critical">
                            <Bug size={14} />
                            <span>{project.criticalIssues} critical issues</span>
                          </div>
                        )}
                        {project.errorAnalyses > 0 && (
                          <div className="alert-item error">
                            <AlertTriangle size={14} />
                            <span>{project.errorAnalyses} error analyses</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Hover Effect */}
                    <div className="project-card-overlay">
                      <div className="overlay-content">
                        <span>View Details</span>
                        <TrendingUp size={16} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default ProjectGrid;
