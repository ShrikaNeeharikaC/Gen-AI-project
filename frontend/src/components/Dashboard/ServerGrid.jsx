// src/components/Dashboard/ServerGrid.jsx - Server selection grid
import React from 'react';
import { Server, Cpu, HardDrive, Wifi, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const ServerGrid = ({ servers, selectedServer, onServerSelect, stage }) => {
  const getServerStatus = (server) => {
    if (!server.health) return 'unknown';
    if (server.health >= 90) return 'healthy';
    if (server.health >= 70) return 'warning';
    return 'critical';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'critical': return XCircle;
      default: return Server;
    }
  };

  const getEnvironmentServers = (environment) => {
    const serverCounts = {
      'dev': 1,
      'qa': 3,
      'stage': 7,
      'production': 10
    };
    return serverCounts[environment] || 1;
  };

  // Generate default servers if none provided
  const defaultServers = [];
  ['dev', 'qa', 'stage', 'production'].forEach(env => {
    const count = getEnvironmentServers(env);
    for (let i = 1; i <= count; i++) {
      defaultServers.push({
        id: `${env}-server-${i}`,
        name: `Server ${i}`,
        environment: env,
        health: Math.floor(Math.random() * 40) + 60, // 60-100%
        cpu_usage: Math.floor(Math.random() * 80) + 10,
        memory_usage: Math.floor(Math.random() * 70) + 20,
        last_deployment: `${Math.floor(Math.random() * 24)} hours ago`,
        status: Math.random() > 0.2 ? 'running' : 'issues'
      });
    }
  });

  const serverData = servers.length > 0 ? servers : defaultServers;

  // Group servers by environment
  const groupedServers = serverData.reduce((acc, server) => {
    const env = server.environment || 'unknown';
    if (!acc[env]) acc[env] = [];
    acc[env].push(server);
    return acc;
  }, {});

  return (
    <div className="server-grid">
      <div className="server-grid-header">
        <h3 className="server-grid-title">
          <Server size={20} />
          Server Infrastructure
          {stage !== 'all' && <span className="stage-context">- {stage} Stage</span>}
        </h3>
        <div className="server-grid-summary">
          <span className="total-servers">
            {serverData.length} servers across {Object.keys(groupedServers).length} environments
          </span>
        </div>
      </div>

      {Object.entries(groupedServers).map(([environment, envServers]) => (
        <div key={environment} className="environment-section">
          <div className="environment-header">
            <h4 className="environment-title">
              {environment.toUpperCase()} Environment
            </h4>
            <div className="environment-stats">
              <span className="server-count">{envServers.length} servers</span>
              <span className="healthy-count">
                {envServers.filter(s => getServerStatus(s) === 'healthy').length} healthy
              </span>
            </div>
          </div>

          <div className="servers-grid">
            {envServers.map((server) => {
              const status = getServerStatus(server);
              const StatusIcon = getStatusIcon(status);
              const isSelected = selectedServer === server.id;

              return (
                <button
                  key={server.id}
                  className={`server-card ${status} ${isSelected ? 'selected' : ''}`}
                  onClick={() => onServerSelect(server.id)}
                >
                  <div className="server-header">
                    <div className="server-info">
                      <div className="server-name">{server.name}</div>
                      <div className="server-id">{server.id}</div>
                    </div>
                    <div className="server-status">
                      <StatusIcon size={16} className="status-icon" />
                    </div>
                  </div>

                  <div className="server-metrics">
                    <div className="metric-row">
                      <div className="metric">
                        <Cpu size={14} />
                        <span className="metric-label">CPU</span>
                        <span className="metric-value">{server.cpu_usage}%</span>
                      </div>
                      <div className="metric">
                        <HardDrive size={14} />
                        <span className="metric-label">Memory</span>
                        <span className="metric-value">{server.memory_usage}%</span>
                      </div>
                    </div>
                    
                    <div className="health-bar">
                      <div className="health-label">Health</div>
                      <div className="health-progress">
                        <div 
                          className={`health-fill ${status}`}
                          style={{ width: `${server.health}%` }}
                        ></div>
                      </div>
                      <div className="health-value">{server.health}%</div>
                    </div>
                  </div>

                  <div className="server-footer">
                    <div className="last-deployment">
                      <Wifi size={12} />
                      <span>Last deploy: {server.last_deployment}</span>
                    </div>
                  </div>

                  {isSelected && (
                    <div className="selection-indicator">
                      <CheckCircle size={16} />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {/* All Servers Option */}
      <div className="all-servers-option">
        <button
          className={`server-card all-servers ${selectedServer === 'all' ? 'selected' : ''}`}
          onClick={() => onServerSelect('all')}
        >
          <div className="all-servers-content">
            <Server size={24} />
            <div className="all-servers-text">
              <div className="all-servers-title">All Servers</div>
              <div className="all-servers-subtitle">View aggregated metrics</div>
            </div>
          </div>
        </button>
      </div>
    </div>
  );
};

export default ServerGrid;
