// src/components/Analytics/AnalysisModal.jsx - Full analysis modal
import React from 'react';
import { X, Copy, Download, ExternalLink, Code, AlertTriangle, CheckCircle, Info } from 'lucide-react';

const AnalysisModal = ({ analysis, onClose }) => {
  const handleCopyAnalysis = () => {
    const analysisText = `
CI/CD Analysis Report
====================
Tool: ${analysis.tool}
Project: ${analysis.project || 'N/A'}
Environment: ${analysis.environment}
Log Type: ${analysis.log_type}
Severity: ${analysis.severity_level}
Timestamp: ${new Date(analysis.analysis_timestamp).toLocaleString()}

Executive Summary:
${analysis.failure_summary || analysis.executive_summary}

Root Cause Analysis:
${analysis.root_cause_analysis || analysis.root_cause}

Fix Strategy:
${analysis.fix_suggestion || analysis.fix_strategy}

Impact Assessment:
${analysis.impact_assessment}

Prevention Measures:
${analysis.prevention_measures}

Monitoring Recommendations:
${analysis.monitoring_recommendations}
    `.trim();

    navigator.clipboard.writeText(analysisText);
  };

  const handleDownload = () => {
    const analysisData = {
      metadata: {
        tool: analysis.tool,
        project: analysis.project,
        environment: analysis.environment,
        log_type: analysis.log_type,
        severity: analysis.severity_level,
        timestamp: analysis.analysis_timestamp,
        confidence: analysis.confidence_score
      },
      analysis: {
        executive_summary: analysis.failure_summary || analysis.executive_summary,
        root_cause: analysis.root_cause_analysis || analysis.root_cause,
        fix_strategy: analysis.fix_suggestion || analysis.fix_strategy,
        impact_assessment: analysis.impact_assessment,
        prevention_measures: analysis.prevention_measures,
        monitoring_recommendations: analysis.monitoring_recommendations
      },
      metrics: {
        error_count: analysis.error_count,
        warning_count: analysis.warning_count,
        success_indicators: analysis.success_indicators,
        business_impact_score: analysis.business_impact_score,
        technical_complexity: analysis.technical_complexity,
        resolution_time_estimate: analysis.resolution_time_estimate
      }
    };

    const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-${analysis.tool}-${analysis.log_type}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

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
        return <Info className="severity-icon info" />;
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="analysis-modal" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title-section">
            <h2 className="modal-title">
              {getSeverityIcon(analysis.severity_level)}
              Complete Analysis Report
            </h2>
            <div className="modal-subtitle">
              {analysis.tool} • {analysis.log_type} • {analysis.environment}
            </div>
          </div>
          
          <div className="modal-actions">
            <button className="action-btn" onClick={handleCopyAnalysis} title="Copy to clipboard">
              <Copy size={16} />
            </button>
            <button className="action-btn" onClick={handleDownload} title="Download as JSON">
              <Download size={16} />
            </button>
            <button className="action-btn close-btn" onClick={onClose} title="Close">
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="modal-content">
          {/* Analysis Overview */}
          <div className="analysis-overview">
            <div className="overview-grid">
              <div className="overview-item">
                <span className="overview-label">Severity Level</span>
                <span className={`overview-value severity-${analysis.severity_level}`}>
                  {analysis.severity_level?.toUpperCase()}
                </span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Confidence Score</span>
                <span className="overview-value">
                  {Math.round((analysis.confidence_score || 0) * 100)}%
                </span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Business Impact</span>
                <span className="overview-value">
                  {analysis.business_impact_score > 0.7 ? 'High' : 
                   analysis.business_impact_score > 0.4 ? 'Medium' : 'Low'}
                </span>
              </div>
              <div className="overview-item">
                <span className="overview-label">Resolution Time</span>
                <span className="overview-value">{analysis.resolution_time_estimate}</span>
              </div>
            </div>
          </div>

          {/* Analysis Sections */}
          <div className="analysis-sections">
            {/* Executive Summary */}
            <div className="analysis-section">
              <h3 className="section-title">
                <Info size={18} />
                Executive Summary
              </h3>
              <div className="section-content">
                {analysis.failure_summary || analysis.executive_summary || 'No summary available'}
              </div>
            </div>

            {/* Root Cause Analysis */}
            <div className="analysis-section">
              <h3 className="section-title">
                <AlertTriangle size={18} />
                Root Cause Analysis
              </h3>
              <div className="section-content">
                {analysis.root_cause_analysis || analysis.root_cause || 'No root cause analysis available'}
              </div>
            </div>

            {/* Fix Strategy */}
            <div className="analysis-section">
              <h3 className="section-title">
                <Code size={18} />
                Fix Strategy
              </h3>
              <div className="section-content">
                <pre className="code-content">
                  {analysis.fix_suggestion || analysis.fix_strategy || 'No fix strategy provided'}
                </pre>
              </div>
            </div>

            {/* Impact Assessment */}
            <div className="analysis-section">
              <h3 className="section-title">
                <ExternalLink size={18} />
                Impact Assessment
              </h3>
              <div className="section-content">
                {analysis.impact_assessment || 'No impact assessment available'}
              </div>
            </div>

            {/* Prevention Measures */}
            <div className="analysis-section">
              <h3 className="section-title">
                <CheckCircle size={18} />
                Prevention Measures
              </h3>
              <div className="section-content">
                {analysis.prevention_measures || 'No prevention measures specified'}
              </div>
            </div>

            {/* Monitoring Recommendations */}
            <div className="analysis-section">
              <h3 className="section-title">
                <AlertTriangle size={18} />
                Monitoring Recommendations
              </h3>
              <div className="section-content">
                {analysis.monitoring_recommendations || 'No monitoring recommendations provided'}
              </div>
            </div>

            {/* Raw LLM Response */}
            {analysis.llm_response && (
              <div className="analysis-section">
                <h3 className="section-title">
                  <Code size={18} />
                  Raw LLM Analysis
                </h3>
                <div className="section-content">
                  <pre className="raw-response">
                    {analysis.llm_response}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <div className="footer-info">
            <span>Analysis generated at {new Date(analysis.analysis_timestamp).toLocaleString()}</span>
            <span>Processing time: {Math.round(analysis.processing_time_ms || 0)}ms</span>
          </div>
          <button className="close-modal-btn" onClick={onClose}>
            Close Analysis
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisModal;
