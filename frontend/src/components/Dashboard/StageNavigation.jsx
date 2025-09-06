// src/components/Dashboard/StageNavigation.jsx - CI/CD stage navigation
import React from 'react';
import { GitBranch, Hammer, TestTube, Shield, Rocket, CheckCircle, XCircle, Clock } from 'lucide-react';

const StageNavigation = ({ stages, selectedStage, onStageSelect }) => {
  const getStageIcon = (stage) => {
    const icons = {
      'all': GitBranch,
      'git_checkout': GitBranch,
      'build': Hammer,
      'test': TestTube,
      'sonarqube': Shield,
      'deployment': Rocket
    };
    return icons[stage] || GitBranch;
  };

  const getStageStatus = (stageData) => {
    if (!stageData) return 'unknown';
    const successRate = stageData.success_rate || 0;
    if (successRate >= 90) return 'success';
    if (successRate >= 70) return 'warning';
    return 'error';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return CheckCircle;
      case 'warning': return Clock;
      case 'error': return XCircle;
      default: return Clock;
    }
  };

  const defaultStages = [
    { name: 'all', label: 'All Stages', success_rate: 85 },
    { name: 'git_checkout', label: 'Source Control', success_rate: 95 },
    { name: 'build', label: 'Build', success_rate: 88 },
    { name: 'test', label: 'Testing', success_rate: 82 },
    { name: 'sonarqube', label: 'Quality Gate', success_rate: 78 },
    { name: 'deployment', label: 'Deployment', success_rate: 91 }
  ];

  const stageData = stages.length > 0 ? stages : defaultStages;

  return (
    <div className="stage-navigation">
      <div className="stage-nav-header">
        <h3 className="stage-nav-title">Pipeline Stages</h3>
        <div className="stage-nav-subtitle">
          Click on a stage to view specific metrics
        </div>
      </div>

      <div className="stage-nav-container">
        <div className="stage-nav-track">
          {stageData.map((stage, index) => {
            const Icon = getStageIcon(stage.name);
            const status = getStageStatus(stage);
            const StatusIcon = getStatusIcon(status);
            const isSelected = selectedStage === stage.name;
            const isLast = index === stageData.length - 1;

            return (
              <div key={stage.name} className="stage-nav-item-wrapper">
                <button
                  className={`stage-nav-item ${isSelected ? 'selected' : ''} ${status}`}
                  onClick={() => onStageSelect(stage.name)}
                >
                  <div className="stage-icon-container">
                    <Icon size={20} className="stage-icon" />
                    <StatusIcon size={12} className="status-icon" />
                  </div>
                  
                  <div className="stage-info">
                    <div className="stage-name">{stage.label}</div>
                    <div className="stage-metrics">
                      <span className="success-rate">
                        {stage.success_rate || 0}% success
                      </span>
                      {stage.last_run && (
                        <span className="last-run">
                          {stage.last_run}
                        </span>
                      )}
                    </div>
                  </div>

                  {isSelected && (
                    <div className="selected-indicator">
                      <div className="indicator-dot"></div>
                    </div>
                  )}
                </button>

                {!isLast && (
                  <div className="stage-connector">
                    <div className="connector-line"></div>
                    <div className="connector-arrow">→</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Stage Summary */}
      <div className="stage-summary">
        <div className="summary-item">
          <span className="summary-label">Total Stages</span>
          <span className="summary-value">{stageData.length - 1}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Healthy Stages</span>
          <span className="summary-value success">
            {stageData.filter(s => getStageStatus(s) === 'success').length - 1}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Issues</span>
          <span className="summary-value error">
            {stageData.filter(s => getStageStatus(s) === 'error').length}
          </span>
        </div>
      </div>
    </div>
  );
};

export default StageNavigation;
