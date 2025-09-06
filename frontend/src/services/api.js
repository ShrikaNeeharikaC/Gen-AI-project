// src/services/api.js - Combined complete version with real data and all helper methods
const API_BASE_URL = 'http://localhost:5001';

class ApiService {
  async request(endpoint, options = {}) {
    try {
      console.log(`Making request to: ${API_BASE_URL}${endpoint}`);
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`Response from ${endpoint}:`, data);
      return data;
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  // UPDATED: Real projects from your cicd_analysis index with enhanced aggregations
  async getProjects() {
    try {
      console.log('Fetching real projects from cicd_analysis index...');
      
      // Query your real cicd_analysis index with comprehensive aggregations
      const projectsResponse = await this.request('/elasticsearch-query', {
        method: 'POST',
        body: JSON.stringify({
          index: 'cicd_analysis',
          query: {
            size: 0,
            aggs: {
              tools: {
                terms: { 
                  field: 'tool',
                  size: 20
                },
                aggs: {
                  projects: {
                    terms: { 
                      field: 'project',
                      size: 50
                    },
                    aggs: {
                      environments: {
                        terms: { field: 'environment' }
                      },
                      latest_analysis: {
                        top_hits: {
                          size: 1,
                          sort: [{ analysis_timestamp: { order: 'desc' } }]
                        }
                      },
                      success_rate: {
                        avg: { 
                          script: {
                            source: "doc['deployment_success'].value ? 1 : 0"
                          }
                        }
                      },
                      total_analyses: {
                        value_count: { field: 'log_id' }
                      },
                      error_analyses: {
                        filter: {
                          range: { error_count: { gt: 0 } }
                        }
                      },
                      critical_issues: {
                        filter: {
                          term: { severity_level: 'critical' }
                        }
                      },
                      avg_confidence: {
                        avg: { field: 'confidence_score' }
                      },
                      avg_build_time: {
                        avg: { field: 'build_duration_seconds' }
                      }
                    }
                  }
                }
              }
            }
          }
        })
      });

      // Transform real Elasticsearch response to project format
      const projects = [];
      
      if (projectsResponse.aggregations?.tools?.buckets) {
        projectsResponse.aggregations.tools.buckets.forEach(toolBucket => {
          const tool = toolBucket.key;
          
          toolBucket.projects.buckets.forEach(projectBucket => {
            const project = projectBucket.key;
            const totalAnalyses = projectBucket.total_analyses.value;
            const errorAnalyses = projectBucket.error_analyses.doc_count;
            const criticalIssues = projectBucket.critical_issues.doc_count;
            const successRate = Math.round((projectBucket.success_rate.value || 0) * 100);
            const avgConfidence = Math.round((projectBucket.avg_confidence.value || 0) * 100);
            const avgBuildTime = Math.round((projectBucket.avg_build_time.value || 0) / 60); // Convert to minutes
            
            // Get latest analysis for last build info
            const latestAnalysis = projectBucket.latest_analysis.hits.hits[0];
            const lastBuild = latestAnalysis ? 
              this.formatLastBuild(latestAnalysis._source.analysis_timestamp) : 'Never';
            
            // Get environment status from environments aggregation
            const envStatus = this.getEnvironmentStatus(projectBucket.environments.buckets, successRate);

            projects.push({
              name: project,
              tool: tool,
              successRate: successRate,
              lastBuild: lastBuild,
              environments: projectBucket.environments.buckets.length,
              envStatus: envStatus,
              totalAnalyses: totalAnalyses,
              errorAnalyses: errorAnalyses,
              criticalIssues: criticalIssues,
              avgConfidence: avgConfidence,
              avgBuildTime: avgBuildTime,
              latestSeverity: latestAnalysis?._source.severity_level || 'unknown'
            });
          });
        });
      }

      console.log(`Found ${projects.length} real projects from database`);
      return projects;
      
    } catch (error) {
      console.error('Failed to get real projects:', error);
      // Return mock data as fallback
      return this.getMockProjects();
    }
  }

  // Get project metrics from ELK database with enhanced functionality
  async getProjectMetrics(tool, project, filters = {}) {
    try {
      const query = this.buildMetricsQuery(tool, project, filters);
      
      const metricsResponse = await this.request('/elasticsearch-query', {
        method: 'POST',
        body: JSON.stringify({
          index: 'cicd_analysis',
          query: query
        })
      });

      return this.transformMetricsResponse(metricsResponse);
    } catch (error) {
      console.error('Failed to get project metrics:', error);
      return this.getMockMetrics();
    }
  }

  // Get project analyses from ELK database with enhanced filtering
  async getProjectAnalyses(tool, project, filters = {}) {
    try {
      const query = this.buildAnalysesQuery(tool, project, filters);
      
      const analysesResponse = await this.request('/elasticsearch-query', {
        method: 'POST',
        body: JSON.stringify({
          index: 'cicd_analysis',
          query: query
        })
      });

      return this.transformAnalysesResponse(analysesResponse);
    } catch (error) {
      console.error('Failed to get project analyses:', error);
      
      // Fallback: get all analyses and filter client-side
      try {
        const allAnalyses = await this.getAllAnalyses();
        return allAnalyses.filter(analysis => 
          analysis.tool === tool && analysis.project === project
        );
      } catch (fallbackError) {
        console.error('Fallback also failed:', fallbackError);
        return this.getMockAnalyses();
      }
    }
  }

  // Get all analyses from database
  async getAllAnalyses() {
    try {
      const response = await this.request('/elasticsearch-query', {
        method: 'POST',
        body: JSON.stringify({
          index: 'cicd_analysis',
          query: {
            query: {
              match_all: {}
            },
            sort: [
              { analysis_timestamp: { order: 'desc' } }
            ],
            size: 100
          }
        })
      });

      if (response.hits && response.hits.hits) {
        return response.hits.hits.map(hit => ({
          id: hit._id,
          ...hit._source
        }));
      }

      return [];
    } catch (error) {
      console.error('Failed to get all analyses:', error);
      throw error;
    }
  }

  // Build ELK query for metrics with comprehensive aggregations
  buildMetricsQuery(tool, project, filters) {
    const mustClauses = [
      { term: { tool: tool } },
      { term: { project: project } }
    ];

    if (filters.stage && filters.stage !== 'all') {
      mustClauses.push({ term: { log_type: filters.stage } });
    }

    if (filters.server && filters.server !== 'all') {
      mustClauses.push({ term: { server: filters.server } });
    }

    if (filters.environment && filters.environment !== 'all') {
      mustClauses.push({ term: { environment: filters.environment } });
    }

    if (filters.timeFilter && filters.timeFilter !== 'all') {
      const timeRange = this.getTimeRange(filters.timeFilter);
      mustClauses.push({
        range: {
          analysis_timestamp: {
            gte: timeRange.start,
            lte: timeRange.end
          }
        }
      });
    }

    return {
      query: {
        bool: { must: mustClauses }
      },
      size: 0,
      aggs: {
        success_rate: {
          avg: { 
            script: {
              source: "doc['deployment_success'].value ? 1 : 0"
            }
          }
        },
        error_rate: {
          avg: { field: 'error_count' }
        },
        avg_build_time: {
          avg: { field: 'build_duration_seconds' }
        },
        deployment_count: {
          value_count: { field: 'deployment_success' }
        },
        success_rate_history: {
          date_histogram: {
            field: 'analysis_timestamp',
            calendar_interval: 'day'
          },
          aggs: {
            success_rate: {
              avg: { 
                script: {
                  source: "doc['deployment_success'].value ? 1 : 0"
                }
              }
            }
          }
        },
        stages_breakdown: {
          terms: { field: 'log_type' },
          aggs: {
            success_rate: {
              avg: { 
                script: {
                  source: "doc['deployment_success'].value ? 1 : 0"
                }
              }
            }
          }
        },
        servers_breakdown: {
          terms: { field: 'server' },
          aggs: {
            health_score: {
              avg: { field: 'confidence_score' }
            }
          }
        },
        log_types_breakdown: {
          terms: { field: 'log_type' },
          aggs: {
            count: {
              value_count: { field: 'log_id' }
            }
          }
        }
      }
    };
  }

  // Build ELK query for analyses with enhanced filtering
  buildAnalysesQuery(tool, project, filters) {
    const mustClauses = [
      { term: { tool: tool } },
      { term: { project: project } }
    ];

    if (filters.stage && filters.stage !== 'all') {
      mustClauses.push({ term: { log_type: filters.stage } });
    }

    if (filters.server && filters.server !== 'all') {
      mustClauses.push({ term: { server: filters.server } });
    }

    if (filters.environment && filters.environment !== 'all') {
      mustClauses.push({ term: { environment: filters.environment } });
    }

    if (filters.logType && filters.logType !== 'all') {
      mustClauses.push({ term: { log_type: filters.logType } });
    }

    if (filters.severity && filters.severity !== 'all') {
      mustClauses.push({ term: { severity_level: filters.severity } });
    }

    if (filters.timeFilter && filters.timeFilter !== 'all') {
      const timeRange = this.getTimeRange(filters.timeFilter);
      mustClauses.push({
        range: {
          analysis_timestamp: {
            gte: timeRange.start,
            lte: timeRange.end
          }
        }
      });
    }

    return {
      query: {
        bool: { must: mustClauses }
      },
      sort: [
        { analysis_timestamp: { order: 'desc' } }
      ],
      size: 50
    };
  }

  // Transform ELK metrics response with real data
  transformMetricsResponse(response) {
    const aggs = response.aggregations || {};
    
    return {
      success_rate: Math.round((aggs.success_rate?.value || 0) * 100),
      error_rate: Math.round(aggs.error_rate?.value || 0),
      avg_build_time: Math.round((aggs.avg_build_time?.value || 0) / 60), // Convert to minutes
      deployment_count: aggs.deployment_count?.value || 0,
      success_rate_history: aggs.success_rate_history?.buckets?.map(bucket => ({
        date: bucket.key_as_string,
        value: Math.round((bucket.success_rate.value || 0) * 100)
      })) || [],
      stages: this.getStagesFromAggregation(aggs.stages_breakdown),
      servers: this.getServersFromAggregation(aggs.servers_breakdown),
      logTypes: this.getLogTypesFromAggregation(aggs.log_types_breakdown)
    };
  }

  // Transform ELK analyses response
  transformAnalysesResponse(response) {
    return response.hits?.hits?.map(hit => ({
      id: hit._id,
      ...hit._source,
      analysis_timestamp: hit._source.analysis_timestamp
    })) || [];
  }

  // Extract stages from real aggregation data
  getStagesFromAggregation(stagesAgg) {
    if (!stagesAgg?.buckets) {
      return this.getStagesFromResponse(); // Fallback to mock data
    }

    const stages = [{ name: 'all', label: 'All Stages', success_rate: 85 }];
    
    stagesAgg.buckets.forEach(bucket => {
      const stageName = bucket.key;
      const successRate = Math.round((bucket.success_rate.value || 0) * 100);
      
      stages.push({
        name: stageName,
        label: this.formatStageLabel(stageName),
        success_rate: successRate
      });
    });

    return stages;
  }

  // Extract servers from real aggregation data
  getServersFromAggregation(serversAgg) {
    if (!serversAgg?.buckets) {
      return this.getServersFromResponse(); // Fallback to mock data
    }

    return serversAgg.buckets.map(bucket => ({
      id: bucket.key,
      name: bucket.key,
      health: Math.round((bucket.health_score.value || 0) * 100)
    }));
  }

  // Extract log types from real aggregation data
  getLogTypesFromAggregation(logTypesAgg) {
    if (!logTypesAgg?.buckets) {
      return this.getLogTypesFromResponse(); // Fallback to mock data
    }

    const logTypes = [{ type: 'all', label: 'All Types', count: 0 }];
    let totalCount = 0;

    logTypesAgg.buckets.forEach(bucket => {
      const count = bucket.count.value;
      totalCount += count;
      
      logTypes.push({
        type: bucket.key,
        label: this.formatLogTypeLabel(bucket.key),
        count: count
      });
    });

    logTypes[0].count = totalCount;
    return logTypes;
  }

  // Helper method to format stage labels
  formatStageLabel(stageName) {
    const stageLabels = {
      'git_checkout': 'Source Control',
      'build': 'Build',
      'test': 'Testing',
      'deployment': 'Deployment',
      'post_deployment': 'Post Deployment'
    };
    return stageLabels[stageName] || stageName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  // Helper method to format log type labels
  formatLogTypeLabel(logType) {
    const logTypeLabels = {
      'build': 'Build Logs',
      'test': 'Test Results',
      'deployment': 'Deployments',
      'error': 'Error Logs'
    };
    return logTypeLabels[logType] || logType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  // Helper method to extract stages from response (fallback)
  getStagesFromResponse(response) {
    return [
      { name: 'all', label: 'All Stages', success_rate: 85 },
      { name: 'git_checkout', label: 'Source Control', success_rate: 95 },
      { name: 'build', label: 'Build', success_rate: 88 },
      { name: 'test', label: 'Testing', success_rate: 82 },
      { name: 'deployment', label: 'Deployment', success_rate: 91 }
    ];
  }

  // Helper method to extract servers from response (fallback)
  getServersFromResponse(response) {
    return [
      { id: 'server-1', name: 'Production Server 1', health: 95 },
      { id: 'server-2', name: 'Production Server 2', health: 88 },
      { id: 'server-3', name: 'Staging Server', health: 92 }
    ];
  }

  // Helper method to extract log types from response (fallback)
  getLogTypesFromResponse(response) {
    return [
      { type: 'all', label: 'All Types', count: 156 },
      { type: 'build', label: 'Build Logs', count: 45 },
      { type: 'test', label: 'Test Results', count: 38 },
      { type: 'deployment', label: 'Deployments', count: 29 }
    ];
  }

  // Helper methods
  getTimeRange(timeFilter) {
    const now = new Date();
    const ranges = {
      '1d': new Date(now.getTime() - 24 * 60 * 60 * 1000),
      '7d': new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
      '30d': new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
    };

    return {
      start: ranges[timeFilter]?.toISOString() || 'now-1y',
      end: now.toISOString()
    };
  }

  formatLastBuild(timestamp) {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  getEnvironmentStatus(envBuckets, overallSuccessRate = 0) {
    const status = {};
    envBuckets.forEach(bucket => {
      const env = bucket.key;
      // Use overall success rate as a baseline for environment status
      status[env] = overallSuccessRate > 80 ? 'success' : 
                   overallSuccessRate > 60 ? 'warning' : 'error';
    });
    return status;
  }

  // Mock data fallbacks
  getMockProjects() {
    return [
      {
        name: 'web-app',
        tool: 'jenkins',
        successRate: 85,
        lastBuild: '2 hours ago',
        environments: 4,
        envStatus: { dev: 'success', qa: 'warning', stage: 'success', production: 'success' },
        totalAnalyses: 156,
        errorAnalyses: 23,
        criticalIssues: 3,
        avgConfidence: 87,
        avgBuildTime: 8,
        latestSeverity: 'medium'
      },
      {
        name: 'api-service',
        tool: 'github_actions',
        successRate: 92,
        lastBuild: '1 hour ago',
        environments: 4,
        envStatus: { dev: 'success', qa: 'success', stage: 'success', production: 'success' },
        totalAnalyses: 203,
        errorAnalyses: 16,
        criticalIssues: 1,
        avgConfidence: 94,
        avgBuildTime: 6,
        latestSeverity: 'low'
      },
      {
        name: 'mobile-app',
        tool: 'gitlab_ci',
        successRate: 78,
        lastBuild: '3 hours ago',
        environments: 3,
        envStatus: { dev: 'success', qa: 'error', production: 'warning' },
        totalAnalyses: 89,
        errorAnalyses: 31,
        criticalIssues: 7,
        avgConfidence: 72,
        avgBuildTime: 12,
        latestSeverity: 'high'
      }
    ];
  }

  getMockMetrics() {
    return {
      success_rate: 85,
      error_rate: 12,
      avg_build_time: 8,
      deployment_count: 45,
      success_rate_history: [
        { date: '2025-06-08', value: 82 },
        { date: '2025-06-09', value: 85 },
        { date: '2025-06-10', value: 88 },
        { date: '2025-06-11', value: 85 }
      ],
      stages: this.getStagesFromResponse(),
      servers: this.getServersFromResponse(),
      logTypes: this.getLogTypesFromResponse()
    };
  }

  getMockAnalyses() {
    return [
      {
        id: 'mock-1',
        tool: 'jenkins',
        project: 'web-app',
        environment: 'production',
        server: 'prod-server-1',
        log_type: 'deployment',
        status: 'error',
        severity_level: 'high',
        confidence_score: 0.92,
        error_count: 3,
        warning_count: 1,
        success_indicators: 0,
        analysis_timestamp: new Date().toISOString(),
        processing_time_ms: 1250,
        failure_summary: 'Docker container failed to start due to port binding conflict',
        root_cause_analysis: 'Port 8080 is already in use by another service',
        fix_suggestion: 'docker stop $(docker ps -q --filter "publish=8080")',
        business_impact_score: 0.85
      }
    ];
  }

  async getStats() {
    return await this.request('/stats');
  }
}

export const apiService = new ApiService();
export default ApiService;