// src/components/Dashboard/MetricsDisplay.jsx - Professional metrics visualization
import React from 'react';
import { TrendingUp, TrendingDown, Activity, Clock, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import MetricChart from '../Analytics/MetricChart';

const MetricsDisplay = ({ metrics, selectedStage, timeFilter }) => {
  const getMetricIcon = (type) => {
    const icons = {
      success_rate: CheckCircle,
      error_rate: XCircle,
      build_time: Clock,
      deployment_frequency: Activity,
      warning_rate: AlertTriangle
    };
    return icons[type] || Activity;
  };

  const getMetricColor = (value, type) => {
    if (type === 'success_rate') {
      return value >= 90 ? 'success' : value >= 70 ? 'warning' : 'error';
    }
    if (type === 'error_rate') {
      return value <= 5 ? 'success' : value <= 15 ? 'warning' : 'error';
    }
    return 'info';
  };

  const formatMetricValue = (value, type) => {
    if (type.includes('rate')) return `${value}%`;
    if (type.includes('time')) return `${value}min`;
    if (type.includes('count')) return value.toString();
    return value;
  };

  const mainMetrics = [
    {
      key: 'success_rate',
      label: 'Success Rate',
      value: metrics.success_rate || 0,
      change: metrics.success_rate_change || 0,
      description: 'Overall pipeline success percentage'
    },
    {
      key: 'error_rate',
      label: 'Error Rate',
      value: metrics.error_rate || 0,
      change: metrics.error_rate_change || 0,
      description: 'Percentage of failed operations'
    },
    {
      key: 'avg_build_time',
      label: 'Avg Build Time',
      value: metrics.avg_build_time || 0,
      change: metrics.build_time_change || 0,
      description: 'Average time for build completion'
    },
    {
      key: 'deployment_frequency',
      label: 'Deployments',
      value: metrics.deployment_count || 0,
      change: metrics.deployment_change || 0,
      description: 'Number of deployments in period'
    }
  ];

  return (
    <div className="metrics-display">
      <div className="metrics-header">
        <h2 className="metrics-title">
          <Activity size={20} />
          Pipeline Metrics
          {selectedStage !== 'all' && (
            <span className="stage-indicator">- {selectedStage}</span>
          )}
        </h2>
        <div className="time-filter-indicator">
          <Clock size={14} />
          <span>{timeFilter === 'all' ? 'All Time' : 
                timeFilter === '30d' ? 'Last 30 Days' :
                timeFilter === '7d' ? 'Last 7 Days' : 'Last 24 Hours'}</span>
        </div>
      </div>

      {/* Main Metrics Grid */}
      <div className="main-metrics-grid">
        {mainMetrics.map((metric) => {
          const Icon = getMetricIcon(metric.key);
          const colorClass = getMetricColor(metric.value, metric.key);
          const isPositiveChange = metric.change >= 0;
          const changeIcon = isPositiveChange ? TrendingUp : TrendingDown;
          const ChangeIcon = changeIcon;

          return (
            <div key={metric.key} className={`metric-card ${colorClass}`}>
              <div className="metric-header">
                <div className="metric-icon">
                  <Icon size={20} />
                </div>
                <div className="metric-change">
                  <ChangeIcon size={14} className={isPositiveChange ? 'positive' : 'negative'} />
                  <span className={isPositiveChange ? 'positive' : 'negative'}>
                    {Math.abs(metric.change)}%
                  </span>
                </div>
              </div>
              
              <div className="metric-content">
                <div className="metric-value">
                  {formatMetricValue(metric.value, metric.key)}
                </div>
                <div className="metric-label">{metric.label}</div>
                <div className="metric-description">{metric.description}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-container">
          <h3 className="chart-title">Success Rate Trend</h3>
          <MetricChart 
            data={metrics.success_rate_history || []}
            type="line"
            color="var(--accent-success)"
          />
        </div>
        
        <div className="chart-container">
          <h3 className="chart-title">Build Time Distribution</h3>
          <MetricChart 
            data={metrics.build_time_distribution || []}
            type="bar"
            color="var(--accent-primary)"
          />
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="additional-metrics">
        <div className="metric-row">
          <div className="metric-item">
            <span className="metric-label">Total Builds</span>
            <span className="metric-value">{metrics.total_builds || 0}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Failed Builds</span>
            <span className="metric-value error">{metrics.failed_builds || 0}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Avg Resolution Time</span>
            <span className="metric-value">{metrics.avg_resolution_time || 'N/A'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Last Deployment</span>
            <span className="metric-value">{metrics.last_deployment || 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsDisplay;
