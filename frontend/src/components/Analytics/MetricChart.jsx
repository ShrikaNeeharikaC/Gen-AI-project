// src/components/Analytics/MetricChart.jsx - Chart component
import React from 'react';

const MetricChart = ({ data, type, color, height = 200 }) => {
  if (!data || data.length === 0) {
    return (
      <div className="chart-placeholder" style={{ height }}>
        <div className="placeholder-content">
          <div className="placeholder-icon">📊</div>
          <div className="placeholder-text">No data available</div>
        </div>
      </div>
    );
  }

  const maxValue = Math.max(...data.map(d => d.value || 0));
  const minValue = Math.min(...data.map(d => d.value || 0));
  const range = maxValue - minValue || 1;

  const renderLineChart = () => {
    const width = 100;
    const chartHeight = height - 40;
    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = chartHeight - ((point.value - minValue) / range) * chartHeight;
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="line-chart" style={{ height }}>
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={color} stopOpacity="0.1" />
            </linearGradient>
          </defs>
          
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(percent => (
            <line
              key={percent}
              x1="0"
              y1={chartHeight * percent / 100}
              x2={width}
              y2={chartHeight * percent / 100}
              stroke="var(--border-secondary)"
              strokeWidth="0.5"
            />
          ))}
          
          {/* Area fill */}
          <polygon
            points={`0,${chartHeight} ${points} ${width},${chartHeight}`}
            fill="url(#gradient)"
          />
          
          {/* Line */}
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
          />
          
          {/* Data points */}
          {data.map((point, index) => {
            const x = (index / (data.length - 1)) * width;
            const y = chartHeight - ((point.value - minValue) / range) * chartHeight;
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="3"
                fill={color}
                className="data-point"
              />
            );
          })}
        </svg>
        
        {/* Labels */}
        <div className="chart-labels">
          <div className="y-axis-labels">
            <span className="label-max">{maxValue}</span>
            <span className="label-min">{minValue}</span>
          </div>
        </div>
      </div>
    );
  };

  const renderBarChart = () => {
    const barWidth = 80 / data.length;
    
    return (
      <div className="bar-chart" style={{ height }}>
        <div className="bars-container">
          {data.map((point, index) => {
            const barHeight = ((point.value - minValue) / range) * (height - 60);
            return (
              <div
                key={index}
                className="bar-item"
                style={{ width: `${barWidth}%` }}
              >
                <div
                  className="bar"
                  style={{
                    height: `${barHeight}px`,
                    backgroundColor: color
                  }}
                  title={`${point.label || index}: ${point.value}`}
                />
                <div className="bar-label">
                  {point.label || index}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="metric-chart">
      {type === 'line' ? renderLineChart() : renderBarChart()}
    </div>
  );
};

export default MetricChart;
