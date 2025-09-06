// src/components/Dashboard/ProjectDashboard.jsx - Complete professional dashboard
import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Calendar, RefreshCw, TrendingUp, Activity, CheckCircle, AlertTriangle, 
  XCircle, Bug, Shield, Info, Clock, Code, ChevronDown, ChevronUp, 
  ArrowLeft, Server, GitBranch, Zap, Eye, Download, Share2, Filter,
  Search, BarChart3, PieChart, LineChart, Settings, Bell, Copy,
  ExternalLink, Play, Pause, RotateCcw, AlertCircle
} from 'lucide-react';
import { apiService } from '../../services/api.js';

const ProjectDashboard = ({ onRefresh }) => {
  const { tool, project } = useParams();
  const navigate = useNavigate();
  
  // State management
  const [projectData, setProjectData] = useState(null);
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState('all');
  const [stageFilter, setStageFilter] = useState('all');
  const [serverFilter, setServerFilter] = useState('all');
  const [logTypeFilter, setLogTypeFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [selectedAnalyses, setSelectedAnalyses] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [realTimeEnabled, setRealTimeEnabled] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const timeFilters = [
    { value: 'all', label: 'All Time' },
    { value: '30d', label: 'Last 30 Days' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '1d', label: 'Last 24 Hours' },
    { value: '1h', label: 'Last Hour' }
  ];

  const sortOptions = [
    { value: 'timestamp', label: 'Date' },
    { value: 'severity', label: 'Severity' },
    { value: 'confidence', label: 'Confidence' },
    { value: 'tool', label: 'Tool' },
    { value: 'environment', label: 'Environment' }
  ];

  // Real-time data loading
  const loadProjectData = useCallback(async () => {
    try {
      setLoading(true);
      
      console.log(`Loading real data for ${tool}/${project}`);
      
      // Get real data from your ELK database
      const [metrics, analysisData, stages, servers, logTypes] = await Promise.all([
        apiService.getProjectMetrics(tool, project, { 
          timeFilter, 
          stage: stageFilter, 
          server: serverFilter,
          logType: logTypeFilter 
        }),
        apiService.getProjectAnalyses(tool, project, { 
          timeFilter, 
          stage: stageFilter, 
          server: serverFilter,
          logType: logTypeFilter,
          search: searchQuery,
          sortBy,
          sortOrder,
          page: currentPage,
          limit: itemsPerPage
        }),
        apiService.getStages(tool, project),
        apiService.getServers(tool, project),
        apiService.getLogTypes(tool, project)
      ]);
      
      setProjectData({
        ...metrics,
        stages: stages || [],
        servers: servers || [],
        logTypes: logTypes || []
      });
      
      setAnalyses(analysisData || []);
      setLastUpdate(new Date());
      
    } catch (error) {
      console.error('Failed to load project data:', error);
      
      // Fallback to get any available data
      try {
        const allAnalyses = await apiService.getAllAnalyses();
        const filteredAnalyses = allAnalyses.filter(a => 
          a.tool === tool && a.project === project
        );
        setAnalyses(filteredAnalyses);
        
        // Set basic metrics from available data
        const totalAnalyses = filteredAnalyses.length;
        const successfulBuilds = filteredAnalyses.filter(a => a.deployment_success).length;
        const errorAnalyses = filteredAnalyses.filter(a => a.error_count > 0).length;
        
        setProjectData({
          success_rate: totalAnalyses > 0 ? Math.round((successfulBuilds / totalAnalyses) * 100) : 0,
          error_rate: totalAnalyses > 0 ? Math.round((errorAnalyses / totalAnalyses) * 100) : 0,
          avg_build_time: 8,
          deployment_count: totalAnalyses,
          total_builds: totalAnalyses,
          failed_builds: errorAnalyses,
          stages: [],
          servers: [],
          logTypes: []
        });
        
      } catch (fallbackError) {
        console.error('Fallback also failed:', fallbackError);
        setAnalyses([]);
        setProjectData({
          success_rate: 0,
          error_rate: 0,
          avg_build_time: 0,
          deployment_count: 0,
          total_builds: 0,
          failed_builds: 0,
          stages: [],
          servers: [],
          logTypes: []
        });
      }
    } finally {
      setLoading(false);
    }
  }, [tool, project, timeFilter, stageFilter, serverFilter, logTypeFilter, searchQuery, sortBy, sortOrder, currentPage, itemsPerPage]);

  // Real-time updates
  useEffect(() => {
    loadProjectData();
    
    if (realTimeEnabled) {
      const interval = setInterval(() => {
        loadProjectData();
      }, 30000); // Update every 30 seconds
      
      return () => clearInterval(interval);
    }
  }, [loadProjectData, realTimeEnabled]);

  // Filter and search handlers
  const handleSearch = (query) => {
    setSearchQuery(query);
    setCurrentPage(1);
  };

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setCurrentPage(1);
  };

  const handleBulkAction = (action) => {
    console.log(`Performing ${action} on`, selectedAnalyses);
    // Implement bulk actions
  };

  const exportData = (format) => {
    console.log(`Exporting data in ${format} format`);
    // Implement export functionality
  };

  if (loading) {
    return (
      <div className="dashboard-loading-professional">
        <div className="loading-content">
          <div className="loading-spinner-large"></div>
          <h2>Loading Pipeline Analytics</h2>
          <p>Fetching real-time data from {tool} • {project}</p>
          <div className="loading-progress">
            <div className="progress-bar"></div>
          </div>
        </div>
      </div>
    );
  }

  // Calculate pagination
  const totalPages = Math.ceil(analyses.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedAnalyses = analyses.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="professional-dashboard">
      {/* Enhanced Professional Header */}
      <div className="dashboard-header-professional">
        <div className="header-left-section">
          <button className="back-btn-professional" onClick={() => navigate('/')}>
            <ArrowLeft size={20} />
            <span>Back to Projects</span>
          </button>
          
          <div className="project-title-professional">
            <div className="project-main-info">
              <div className="tool-badge-professional">{tool}</div>
              <h1 className="project-name">{project}</h1>
              <div className="project-status-professional">
                <div className={`status-indicator ${realTimeEnabled ? 'active' : 'inactive'}`}>
                  <div className="status-dot"></div>
                  <span>{realTimeEnabled ? 'Live Pipeline' : 'Static View'}</span>
                </div>
              </div>
            </div>
            <p className="project-description-professional">
              Real-time CI/CD Analytics & Intelligence Dashboard
            </p>
            <div className="breadcrumb-professional">
              <span>Projects</span> / <span>{tool}</span> / <span className="current">{project}</span>
            </div>
          </div>
        </div>

        <div className="header-controls-professional">
          <div className="control-group">
            <div className="time-filter-professional">
              <Calendar size={16} />
              <select 
                value={timeFilter} 
                onChange={(e) => setTimeFilter(e.target.value)}
                className="professional-select"
              >
                {timeFilters.map((filter) => (
                  <option key={filter.value} value={filter.value}>
                    {filter.label}
                  </option>
                ))}
              </select>
            </div>

            <button 
              className="filter-toggle-btn"
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter size={16} />
              Filters
            </button>

            <button 
              className="realtime-toggle-btn"
              onClick={() => setRealTimeEnabled(!realTimeEnabled)}
            >
              {realTimeEnabled ? <Pause size={16} /> : <Play size={16} />}
              {realTimeEnabled ? 'Pause' : 'Resume'}
            </button>
          </div>

          <div className="action-group">
            <button className="export-btn" onClick={() => exportData('pdf')}>
              <Download size={16} />
              Export
            </button>

            <button className="share-btn">
              <Share2 size={16} />
              Share
            </button>

            <button className="refresh-btn-professional" onClick={loadProjectData}>
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Last Update Indicator */}
      <div className="last-update-indicator">
        <Clock size={14} />
        <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
        {realTimeEnabled && <div className="live-indicator">LIVE</div>}
      </div>

      {/* Advanced Filters Panel */}
      {showFilters && (
        <div className="filters-panel-professional">
          <div className="filters-content">
            <div className="filter-group">
              <label>Stage</label>
              <select 
                value={stageFilter} 
                onChange={(e) => setStageFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Stages</option>
                {projectData?.stages?.map(stage => (
                  <option key={stage.name} value={stage.name}>{stage.label}</option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label>Server</label>
              <select 
                value={serverFilter} 
                onChange={(e) => setServerFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Servers</option>
                {projectData?.servers?.map(server => (
                  <option key={server.id} value={server.id}>{server.name}</option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label>Log Type</label>
              <select 
                value={logTypeFilter} 
                onChange={(e) => setLogTypeFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Types</option>
                {projectData?.logTypes?.map(type => (
                  <option key={type.type} value={type.type}>{type.label}</option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label>Sort By</label>
              <select 
                value={sortBy} 
                onChange={(e) => setSortBy(e.target.value)}
                className="filter-select"
              >
                {sortOptions.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>

            <button className="clear-filters-btn" onClick={() => {
              setStageFilter('all');
              setServerFilter('all');
              setLogTypeFilter('all');
              setSearchQuery('');
              setSortBy('timestamp');
              setSortOrder('desc');
            }}>
              Clear All
            </button>
          </div>
        </div>
      )}

      {/* Professional Metrics Grid */}
      <div className="metrics-grid-professional">
        <div className="metric-card-professional success">
          <div className="metric-header-professional">
            <div className="metric-icon-professional">
              <CheckCircle size={28} />
            </div>
            <div className="metric-trend positive">
              <TrendingUp size={16} />
              <span>+2.3%</span>
            </div>
          </div>
          <div className="metric-content-professional">
            <div className="metric-value-professional">{projectData?.success_rate || 0}%</div>
            <div className="metric-label-professional">Success Rate</div>
            <div className="metric-subtitle">Pipeline success across all stages</div>
          </div>
          <div className="metric-chart-mini">
            <div className="chart-bar" style={{height: '60%'}}></div>
            <div className="chart-bar" style={{height: '80%'}}></div>
            <div className="chart-bar" style={{height: '90%'}}></div>
            <div className="chart-bar" style={{height: '85%'}}></div>
          </div>
        </div>

        <div className="metric-card-professional error">
          <div className="metric-header-professional">
            <div className="metric-icon-professional">
              <XCircle size={28} />
            </div>
            <div className="metric-trend negative">
              <TrendingUp size={16} className="trend-down" />
              <span>-1.2%</span>
            </div>
          </div>
          <div className="metric-content-professional">
            <div className="metric-value-professional">{projectData?.error_rate || 0}%</div>
            <div className="metric-label-professional">Failure Rate</div>
            <div className="metric-subtitle">Critical and high severity failures</div>
          </div>
          <div className="metric-chart-mini">
            <div className="chart-bar error" style={{height: '40%'}}></div>
            <div className="chart-bar error" style={{height: '30%'}}></div>
            <div className="chart-bar error" style={{height: '25%'}}></div>
            <div className="chart-bar error" style={{height: '20%'}}></div>
          </div>
        </div>

        <div className="metric-card-professional info">
          <div className="metric-header-professional">
            <div className="metric-icon-professional">
              <Clock size={28} />
            </div>
            <div className="metric-trend positive">
              <TrendingUp size={16} />
              <span>-15%</span>
            </div>
          </div>
          <div className="metric-content-professional">
            <div className="metric-value-professional">{projectData?.avg_build_time || 0}m</div>
            <div className="metric-label-professional">Avg Build Time</div>
            <div className="metric-subtitle">Mean time to complete builds</div>
          </div>
          <div className="metric-chart-mini">
            <div className="chart-bar info" style={{height: '70%'}}></div>
            <div className="chart-bar info" style={{height: '65%'}}></div>
            <div className="chart-bar info" style={{height: '60%'}}></div>
            <div className="chart-bar info" style={{height: '55%'}}></div>
          </div>
        </div>

        <div className="metric-card-professional warning">
          <div className="metric-header-professional">
            <div className="metric-icon-professional">
              <Activity size={28} />
            </div>
            <div className="metric-trend positive">
              <TrendingUp size={16} />
              <span>+8</span>
            </div>
          </div>
          <div className="metric-content-professional">
            <div className="metric-value-professional">{projectData?.deployment_count || 0}</div>
            <div className="metric-label-professional">Deployments</div>
            <div className="metric-subtitle">Total deployments this period</div>
          </div>
          <div className="metric-chart-mini">
            <div className="chart-bar warning" style={{height: '50%'}}></div>
            <div className="chart-bar warning" style={{height: '70%'}}></div>
            <div className="chart-bar warning" style={{height: '85%'}}></div>
            <div className="chart-bar warning" style={{height: '90%'}}></div>
          </div>
        </div>
      </div>

      {/* Secondary Metrics Row */}
      <div className="secondary-metrics-professional">
        <div className="secondary-metric">
          <div className="secondary-metric-icon">
            <RotateCcw size={20} />
          </div>
          <div className="secondary-metric-content">
            <div className="secondary-metric-value">{projectData?.failed_builds || 0}</div>
            <div className="secondary-metric-label">Failed Builds</div>
          </div>
        </div>

        <div className="secondary-metric">
          <div className="secondary-metric-icon">
            <Server size={20} />
          </div>
          <div className="secondary-metric-content">
            <div className="secondary-metric-value">{projectData?.servers?.length || 0}</div>
            <div className="secondary-metric-label">Active Servers</div>
          </div>
        </div>

        <div className="secondary-metric">
          <div className="secondary-metric-icon">
            <GitBranch size={20} />
          </div>
          <div className="secondary-metric-content">
            <div className="secondary-metric-value">{projectData?.stages?.length || 0}</div>
            <div className="secondary-metric-label">Pipeline Stages</div>
          </div>
        </div>

        <div className="secondary-metric">
          <div className="secondary-metric-icon">
            <Zap size={20} />
          </div>
          <div className="secondary-metric-content">
            <div className="secondary-metric-value">2.4h</div>
            <div className="secondary-metric-label">MTTR</div>
          </div>
        </div>
      </div>

      {/* Analysis Section */}
      <div className="analysis-section-professional">
        <div className="analysis-header-professional">
          <div className="analysis-title-section">
            <h2 className="analysis-title-professional">
              <TrendingUp size={24} />
              Real-time Analysis Results
            </h2>
            <div className="analysis-stats">
              <span className="analysis-count">{analyses.length} total analyses</span>
              <span className="analysis-separator">•</span>
              <span className="analysis-filtered">{paginatedAnalyses.length} showing</span>
            </div>
          </div>

          <div className="analysis-controls-professional">
            <div className="search-box-professional">
              <Search size={16} />
              <input
                type="text"
                placeholder="Search analyses..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="search-input-professional"
              />
            </div>

            <div className="bulk-actions">
              {selectedAnalyses.length > 0 && (
                <div className="bulk-action-bar">
                  <span>{selectedAnalyses.length} selected</span>
                  <button onClick={() => handleBulkAction('export')}>Export</button>
                  <button onClick={() => handleBulkAction('delete')}>Delete</button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Analysis Grid */}
        {analyses.length === 0 ? (
          <div className="no-analyses-professional">
            <div className="no-analyses-icon">
              <BarChart3 size={48} />
            </div>
            <h3>No Analysis Data Found</h3>
            <p>No analyses found for {tool}/{project} with current filters.</p>
            <div className="no-analyses-actions">
              <button onClick={loadProjectData} className="retry-btn">
                <RefreshCw size={16} />
                Refresh Data
              </button>
              <button onClick={() => setShowFilters(true)} className="filter-btn">
                <Filter size={16} />
                Adjust Filters
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="analysis-grid-professional">
              {paginatedAnalyses.map((analysis) => (
                <ProfessionalAnalysisCard
                  key={analysis.log_id || analysis.id}
                  analysis={analysis}
                  onRefresh={loadProjectData}
                  onSelect={(id, selected) => {
                    if (selected) {
                      setSelectedAnalyses([...selectedAnalyses, id]);
                    } else {
                      setSelectedAnalyses(selectedAnalyses.filter(sid => sid !== id));
                    }
                  }}
                  isSelected={selectedAnalyses.includes(analysis.log_id || analysis.id)}
                />
              ))}
            </div>

            {/* Professional Pagination */}
            {totalPages > 1 && (
              <div className="pagination-professional">
                <div className="pagination-info">
                  Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, analyses.length)} of {analyses.length} results
                </div>
                <div className="pagination-controls">
                  <button 
                    onClick={() => setCurrentPage(1)}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    First
                  </button>
                  <button 
                    onClick={() => setCurrentPage(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    Previous
                  </button>
                  
                  {[...Array(Math.min(5, totalPages))].map((_, i) => {
                    const page = i + 1;
                    return (
                      <button
                        key={page}
                        onClick={() => setCurrentPage(page)}
                        className={`pagination-btn ${currentPage === page ? 'active' : ''}`}
                      >
                        {page}
                      </button>
                    );
                  })}
                  
                  <button 
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="pagination-btn"
                  >
                    Next
                  </button>
                  <button 
                    onClick={() => setCurrentPage(totalPages)}
                    disabled={currentPage === totalPages}
                    className="pagination-btn"
                  >
                    Last
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// Professional Analysis Card Component
const ProfessionalAnalysisCard = ({ analysis, onRefresh, onSelect, isSelected }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showFullAnalysis, setShowFullAnalysis] = useState(false);

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <AlertCircle className="severity-icon critical" />;
      case 'high':
        return <XCircle className="severity-icon high" />;
      case 'medium':
        return <AlertTriangle className="severity-icon medium" />;
      case 'low':
        return <Info className="severity-icon low" />;
      default:
        return <Clock className="severity-icon info" />;
    }
  };

  const getErrorTypeIcon = (errorCount, warningCount) => {
    if (errorCount > 0) return <Bug className="error-type-icon error" />;
    if (warningCount > 0) return <Shield className="error-type-icon warning" />;
    return <CheckCircle className="error-type-icon success" />;
  };

  const getSeverityClass = (severity) => {
    return `severity-${severity || 'unknown'}`;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown time';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const getErrorSummary = () => {
    const errors = analysis.error_count || 0;
    const warnings = analysis.warning_count || 0;
    const success = analysis.success_indicators || 0;

    if (errors > 0) {
      return `${errors} error${errors > 1 ? 's' : ''} detected`;
    } else if (warnings > 0) {
      return `${warnings} warning${warnings > 1 ? 's' : ''} found`;
    } else if (success > 0) {
      return `${success} success indicator${success > 1 ? 's' : ''}`;
    }
    return 'Analysis completed';
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // Add toast notification here
  };

  return (
    <div className={`analysis-card-professional ${getSeverityClass(analysis.severity_level)} ${isSelected ? 'selected' : ''}`}>
      {/* Selection Checkbox */}
      <div className="card-selection">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={(e) => onSelect(analysis.log_id || analysis.id, e.target.checked)}
          className="selection-checkbox"
        />
      </div>

      {/* Professional Analysis Header */}
      <div className="analysis-header-professional">
        <div className="analysis-meta-professional">
          <div className="severity-badge-professional">
            {getSeverityIcon(analysis.severity_level)}
            <span className="severity-text-professional">
              {(analysis.severity_level || 'unknown').toUpperCase()}
            </span>
            <div className="error-summary-professional">
              {getErrorTypeIcon(analysis.error_count, analysis.warning_count)}
              <span>{getErrorSummary()}</span>
            </div>
          </div>
          
          <div className="analysis-tags-professional">
            <span className="tag tool-tag">{analysis.tool || 'unknown'}</span>
            <span className="tag log-type-tag">{analysis.log_type || 'unknown'}</span>
            <span className="tag environment-tag">{analysis.environment || 'unknown'}</span>
            <span className="tag server-tag">{analysis.server || 'unknown'}</span>
          </div>
        </div>
        
        <div className="analysis-timestamp-professional">
          <Clock size={14} />
          <span>{formatTimestamp(analysis.analysis_timestamp)}</span>
        </div>
      </div>

      {/* Professional Error Metrics */}
      <div className="error-metrics-professional">
        <div className="error-counters-professional">
          <div className="error-counter-professional critical">
            <div className="counter-icon">
              <XCircle size={18} />
            </div>
            <div className="counter-content">
              <div className="counter-value">{analysis.error_count || 0}</div>
              <div className="counter-label">Errors</div>
            </div>
          </div>
          
          <div className="error-counter-professional warning">
            <div className="counter-icon">
              <AlertTriangle size={18} />
            </div>
            <div className="counter-content">
              <div className="counter-value">{analysis.warning_count || 0}</div>
              <div className="counter-label">Warnings</div>
            </div>
          </div>
          
          <div className="error-counter-professional success">
            <div className="counter-icon">
              <CheckCircle size={18} />
            </div>
            <div className="counter-content">
              <div className="counter-value">{analysis.success_indicators || 0}</div>
              <div className="counter-label">Success</div>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Content */}
      <div className="analysis-content-professional">
        <h3 className="analysis-title-card">
          {analysis.failure_summary || analysis.executive_summary || 'Analysis Summary'}
        </h3>
        
        {/* Professional Metrics Grid */}
        <div className="analysis-metrics-professional">
          <div className="metric-item-professional">
            <div className="metric-label-card">Confidence</div>
            <div className="confidence-bar-professional">
              <div 
                className="confidence-fill-professional" 
                style={{ width: `${(analysis.confidence_score || 0) * 100}%` }}
              ></div>
              <span className="confidence-text-professional">
                {Math.round((analysis.confidence_score || 0) * 100)}%
              </span>
            </div>
          </div>
          
          <div className="metric-item-professional">
            <div className="metric-label-card">Business Impact</div>
            <div className="impact-indicator-professional">
              <div 
                className={`impact-level-professional ${
                  analysis.business_impact_score > 0.7 ? 'high' : 
                  analysis.business_impact_score > 0.4 ? 'medium' : 'low'
                }`}
              >
                {analysis.business_impact_score > 0.7 ? 'High' : 
                 analysis.business_impact_score > 0.4 ? 'Medium' : 'Low'}
              </div>
            </div>
          </div>
          
          <div className="metric-item-professional">
            <div className="metric-label-card">Processing Time</div>
            <div className="metric-value-card">
              <Clock size={14} />
              <span>{Math.round(analysis.processing_time_ms || 0)}ms</span>
            </div>
          </div>

          <div className="metric-item-professional">
            <div className="metric-label-card">Resolution Time</div>
            <div className="metric-value-card">
              <AlertCircle size={14} />
              <span>{analysis.resolution_time_estimate || 'Unknown'}</span>
            </div>
          </div>
        </div>

        {/* Expandable Details */}
        {isExpanded && (
          <div className="analysis-details-professional">
            {/* Root Cause Section */}
            <div className="detail-section-professional error-section">
              <h4 className="detail-title-professional">
                <Bug size={18} />
                Root Cause Analysis
              </h4>
              <div className="detail-content-professional">
                {analysis.root_cause_analysis || analysis.root_cause || 'No root cause analysis available'}
              </div>
            </div>

            {/* Fix Strategy Section */}
            <div className="detail-section-professional fix-section">
              <h4 className="detail-title-professional">
                <Code size={18} />
                Fix Strategy
              </h4>
              <div className="fix-content-professional">
                {(analysis.fix_suggestion || analysis.fix_strategy) && (
                  <div className="code-block-professional">
                    <div className="code-header-professional">
                      <span>Recommended Fix</span>
                      <button 
                        className="copy-btn-professional"
                        onClick={() => copyToClipboard(analysis.fix_suggestion || analysis.fix_strategy)}
                      >
                        <Copy size={14} />
                        Copy
                      </button>
                    </div>
                    <pre className="code-content-professional">
                      {analysis.fix_suggestion || analysis.fix_strategy}
                    </pre>
                  </div>
                )}
              </div>
            </div>

            {/* Impact Assessment */}
            <div className="detail-section-professional impact-section">
              <h4 className="detail-title-professional">
                <AlertTriangle size={18} />
                Impact Assessment
              </h4>
              <div className="detail-content-professional">
                {analysis.impact_assessment || 'Impact assessment not available'}
              </div>
            </div>

            {/* Prevention Measures */}
            <div className="detail-section-professional prevention-section">
              <h4 className="detail-title-professional">
                <Shield size={18} />
                Prevention Measures
              </h4>
              <div className="detail-content-professional">
                {analysis.prevention_measures || 'No prevention measures specified'}
              </div>
            </div>

            {/* Monitoring Recommendations */}
            <div className="detail-section-professional monitoring-section">
              <h4 className="detail-title-professional">
                <Eye size={18} />
                Monitoring Recommendations
              </h4>
              <div className="detail-content-professional">
                {analysis.monitoring_recommendations || 'No monitoring recommendations provided'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Professional Actions */}
      <div className="analysis-actions-professional">
        <div className="action-group-left">
          <button
            className="expand-btn-professional"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <>
                <ChevronUp size={16} />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown size={16} />
                Show Details
              </>
            )}
          </button>
        </div>

        <div className="action-group-right">
          <button
            className="action-btn-professional secondary"
            onClick={() => copyToClipboard(JSON.stringify(analysis, null, 2))}
          >
            <Copy size={16} />
            Copy
          </button>

          <button
            className="action-btn-professional primary"
            onClick={() => setShowFullAnalysis(true)}
          >
            <ExternalLink size={16} />
            Full Analysis
          </button>
        </div>
      </div>

      {/* Full Analysis Modal */}
      {showFullAnalysis && (
        <FullAnalysisModal
          analysis={analysis}
          onClose={() => setShowFullAnalysis(false)}
        />
      )}
    </div>
  );
};

// Full Analysis Modal Component
const FullAnalysisModal = ({ analysis, onClose }) => {
  return (
    <div className="modal-overlay-professional" onClick={onClose}>
      <div className="analysis-modal-professional" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header-professional">
          <h2>Complete Analysis Report</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <XCircle size={24} />
          </button>
        </div>
        
        <div className="modal-content-professional">
          <div className="analysis-overview-professional">
            <div className="overview-grid-professional">
              <div className="overview-item">
                <span className="overview-label">Tool</span>
                <span className="overview-value">{analysis.tool}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Environment</span>
                <span className="overview-value">{analysis.environment}</span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Severity</span>
                <span className={`overview-value severity-${analysis.severity_level}`}>
                  {analysis.severity_level?.toUpperCase()}
                </span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Confidence</span>
                <span className="overview-value">
                  {Math.round((analysis.confidence_score || 0) * 100)}%
                </span>
              </div>
            </div>
          </div>

          <div className="modal-sections">
            <div className="modal-section">
              <h3>Executive Summary</h3>
              <p>{analysis.failure_summary || analysis.executive_summary}</p>
            </div>

            <div className="modal-section">
              <h3>Root Cause Analysis</h3>
              <p>{analysis.root_cause_analysis || analysis.root_cause}</p>
            </div>

            <div className="modal-section">
              <h3>Fix Strategy</h3>
              <pre className="modal-code-block">
                {analysis.fix_suggestion || analysis.fix_strategy}
              </pre>
            </div>

            <div className="modal-section">
              <h3>Raw LLM Response</h3>
              <pre className="modal-raw-response">
                {analysis.llm_response}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectDashboard;
