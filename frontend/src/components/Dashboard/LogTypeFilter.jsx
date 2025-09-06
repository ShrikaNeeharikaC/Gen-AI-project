// src/components/Dashboard/LogTypeFilter.jsx - Log type filtering
import React from 'react';
import { Filter, FileText, GitBranch, Hammer, TestTube, Shield, Rocket } from 'lucide-react';

const LogTypeFilter = ({ logTypes, selectedLogType, onLogTypeSelect }) => {
  const getLogTypeIcon = (type) => {
    const icons = {
      'all': FileText,
      'git_checkout': GitBranch,
      'build': Hammer,
      'test': TestTube,
      'sonarqube': Shield,
      'deployment': Rocket
    };
    return icons[type] || FileText;
  };

  const getLogTypeColor = (type) => {
    const colors = {
      'git_checkout': 'git',
      'build': 'build',
      'test': 'test',
      'sonarqube': 'quality',
      'deployment': 'deploy'
    };
    return colors[type] || 'default';
  };

  const defaultLogTypes = [
    { 
      type: 'all', 
      label: 'All Log Types', 
      count: 156, 
      description: 'View all log entries across all types',
      recent_activity: '2 min ago'
    },
    { 
      type: 'git_checkout', 
      label: 'Source Control', 
      count: 45, 
      description: 'Git operations, checkout, and version control',
      recent_activity: '5 min ago'
    },
    { 
      type: 'build', 
      label: 'Build Logs', 
      count: 38, 
      description: 'Compilation, packaging, and build processes',
      recent_activity: '3 min ago'
    },
    { 
      type: 'test', 
      label: 'Test Results', 
      count: 29, 
      description: 'Unit tests, integration tests, and coverage',
      recent_activity: '7 min ago'
    },
    { 
      type: 'sonarqube', 
      label: 'Quality Gates', 
      count: 22, 
      description: 'Code quality analysis and security scans',
      recent_activity: '12 min ago'
    },
    { 
      type: 'deployment', 
      label: 'Deployments', 
      count: 22, 
      description: 'Application deployment and release logs',
      recent_activity: '8 min ago'
    }
  ];

  const logTypeData = logTypes.length > 0 ? logTypes : defaultLogTypes;

  return (
    <div className="log-type-filter">
      <div className="filter-header">
        <h3 className="filter-title">
          <Filter size={20} />
          Log Type Filter
        </h3>
        <div className="filter-subtitle">
          Filter analysis results by log type
        </div>
      </div>

      <div className="log-type-grid">
        {logTypeData.map((logType) => {
          const Icon = getLogTypeIcon(logType.type);
          const colorClass = getLogTypeColor(logType.type);
          const isSelected = selectedLogType === logType.type;

          return (
            <button
              key={logType.type}
              className={`log-type-card ${colorClass} ${isSelected ? 'selected' : ''}`}
              onClick={() => onLogTypeSelect(logType.type)}
            >
              <div className="log-type-header">
                <div className="log-type-icon">
                  <Icon size={20} />
                </div>
                <div className="log-type-count">
                  {logType.count}
                </div>
              </div>

              <div className="log-type-content">
                <div className="log-type-label">{logType.label}</div>
                <div className="log-type-description">{logType.description}</div>
              </div>

              <div className="log-type-footer">
                <div className="recent-activity">
                  <span className="activity-dot"></span>
                  <span className="activity-text">Last: {logType.recent_activity}</span>
                </div>
              </div>

              {isSelected && (
                <div className="selection-overlay">
                  <div className="selection-checkmark">✓</div>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Quick Stats */}
      <div className="log-type-stats">
        <div className="stats-row">
          <div className="stat-item">
            <span className="stat-label">Total Logs</span>
            <span className="stat-value">
              {logTypeData.reduce((sum, type) => sum + (type.count || 0), 0)}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Active Types</span>
            <span className="stat-value">
              {logTypeData.filter(type => type.count > 0).length - 1}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Most Recent</span>
            <span className="stat-value">
              {logTypeData.find(type => type.type !== 'all')?.recent_activity || 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LogTypeFilter;
