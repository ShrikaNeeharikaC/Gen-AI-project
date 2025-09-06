// src/components/Analytics/AnalysisCard.jsx - Beautiful analysis display
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Clock, Code } from 'lucide-react';
import AnalysisModal from './AnalysisModal';

const AnalysisCard = ({ analysis, onRefresh }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical':
      case 'high':
        return <AlertTriangle className="severity-icon critical" />;
      case 'medium':
        return <AlertTriangle className="severity-icon warning" />;
      case 'low':
        return <CheckCircle className="severity-icon success" />;
      default:
        return <Clock className="severity-icon info" />;
    }
  };

  const getSeverityClass = (severity) => {
    return `severity-${severity}`;
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <>
      <div className={`analysis-card ${getSeverityClass(analysis.severity_level)}`}>
        <div className="analysis-header">
          <div className="analysis-meta">
            <div className="severity-badge">
              {getSeverityIcon(analysis.severity_level)}
              <span className="severity-text">{analysis.severity_level}</span>
            </div>
            <div className="analysis-info">
              <span className="tool-tag">{analysis.tool}</span>
              <span className="log-type-tag">{analysis.log_type}</span>
              <span className="environment-tag">{analysis.environment}</span>
            </div>
          </div>
          <div className="analysis-timestamp">
            <Clock size={14} />
            {formatTimestamp(analysis.analysis_timestamp)}
          </div>
        </div>

        <div className="analysis-content">
          <h3 className="analysis-title">
            {analysis.failure_summary || 'Analysis Summary'}
          </h3>
          
          <div className="analysis-metrics">
            <div className="metric-item">
              <span className="metric-label">Confidence</span>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${(analysis.confidence_score || 0) * 100}%` }}
                ></div>
                <span className="confidence-text">
                  {Math.round((analysis.confidence_score || 0) * 100)}%
                </span>
              </div>
            </div>
            
            <div className="metric-item">
              <span className="metric-label">Processing Time</span>
              <span className="metric-value">
                {Math.round(analysis.processing_time_ms || 0)}ms
              </span>
            </div>
          </div>

          {isExpanded && (
            <div className="analysis-details">
              <div className="detail-section">
                <h4 className="detail-title">Root Cause</h4>
                <p className="detail-content">
                  {analysis.root_cause_analysis || 'No root cause analysis available'}
                </p>
              </div>

              <div className="detail-section">
                <h4 className="detail-title">Fix Strategy</h4>
                <div className="fix-content">
                  {analysis.fix_suggestion && (
                    <pre className="code-block">
                      <Code size={16} />
                      {analysis.fix_suggestion}
                    </pre>
                  )}
                </div>
              </div>

              <div className="detail-section">
                <h4 className="detail-title">Impact Assessment</h4>
                <p className="detail-content">
                  {analysis.impact_assessment || 'Impact assessment not available'}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="analysis-actions">
          <button
            className="expand-btn"
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
                Show More
              </>
            )}
          </button>
          
          <button
            className="full-analysis-btn"
            onClick={() => setShowModal(true)}
          >
            Show Full Analysis
          </button>
        </div>
      </div>

      {showModal && (
        <AnalysisModal
          analysis={analysis}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
};

export default AnalysisCard;
