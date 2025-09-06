from flask import Flask, jsonify, request
from flask_cors import CORS
from elasticsearch import Elasticsearch
import logging
from datetime import datetime
import re, json

def parse_full_synthesis(synthesis):
    """Safely parse full_synthesis field"""
    if synthesis:
        try:
            if isinstance(synthesis, dict):
                return synthesis
            elif isinstance(synthesis, str):
                return json.loads(synthesis)
            else:
                return {}
        except Exception as e:
            return {}
    return {}

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch configuration
es = Elasticsearch([
    'https://a705a31d6c434d5d9b8801b99d0ef7f7.us-central1.gcp.cloud.es.io'
], 
api_key='SEtWQlVKY0JRLUE2QldTNnB3c0U6TXJ6a1dKZ0xIQ01fTndYNWtLRVhhdw==')

analysis_index = 'cicd_analysis'

@app.route('/health', methods=['GET'])
def health_check():
    try:
        es.ping()
        return jsonify({"status": "healthy", "elasticsearch": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/projects', methods=['GET'])
def get_projects():
    """Get all projects grouped by tool"""
    try:
        body = {
            "size": 0,
            "aggs": {
                "tools": {
                    "terms": {"field": "tool", "size": 20},
                    "aggs": {
                        "projects": {
                            "terms": {"field": "project", "size": 100}
                        }
                    }
                }
            }
        }
        
        response = es.search(index=analysis_index, body=body)

        print("Raw elasticsearch response:", response)  # Debugging line

        tools_data = []
        
        if 'aggregations' not in response:
            print("No aggregations found in response")
            return jsonify([])


        for tool_bucket in response['aggregations']['tools']['buckets']:
            projects = []
            for project_bucket in tool_bucket['projects']['buckets']:
                projects.append({
                    "name": project_bucket['key'],
                    "doc_count": project_bucket['doc_count']
                })
            
            tools_data.append({
                "tool": tool_bucket['key'],
                "projects": projects,
                "total_builds": tool_bucket['doc_count']
            })
        print("Formatted tools data:", tools_data)  # Debugging line
        return jsonify(tools_data)
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        print(f"Exception details: {e}")  # Debugging line
        return jsonify({"error": str(e)}), 500

@app.route('/project-metrics/<tool>/<project>', methods=['GET'])
def get_project_metrics(tool, project):
    """Get comprehensive metrics for a specific project"""
    try:
        # Main metrics query
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tool": tool}},
                        {"term": {"project": project}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "total_builds": {"value_count": {"field": "build_duration_seconds"}},
                "successful_builds": {"filter": {"term": {"deployment_success": True}}},
                "failed_builds": {"filter": {"term": {"deployment_success": False}}},
                "avg_build_time": {"avg": {"field": "build_duration_seconds"}},
                "total_errors": {"filter": {"term": {"status": "error"}}},
                "processing_times": {"terms": {"field": "processing_time_ms", "size": 1000}},
                "recent_builds": {
                    "date_histogram": {
                        "field": "analysis_timestamp",
                        "calendar_interval": "1d",
                        "min_doc_count": 0
                    },
                    "aggs": {
                        "success_rate": {
                            "bucket_script": {
                                "buckets_path": {
                                    "total": "_count",
                                    "successful": "successful_builds>_count"
                                },
                                "script": "params.total > 0 ? (params.successful / params.total) * 100 : 0"
                            }
                        },
                        "successful_builds": {"filter": {"term": {"deployment_success": True}}}
                    }
                }
            }
        }
        
        response = es.search(index=analysis_index, body=body)
        aggs = response.get('aggregations', {})
        
        # Calculate basic metrics
        total_builds = aggs.get('total_builds', {}).get('value', 0) or 0
        successful_builds = aggs.get('successful_builds', {}).get('doc_count', 0) or 0
        failed_builds = aggs.get('failed_builds', {}).get('doc_count', 0) or 0
        avg_build_time = aggs.get('avg_build_time', {}).get('value', 0) or 0
        total_errors = aggs.get('total_errors', {}).get('doc_count', 0) or 0
        
        # Calculate rates
        success_rate = (successful_builds / total_builds) * 100 if total_builds > 0 else 0
        failure_rate = (failed_builds / total_builds) * 100 if total_builds > 0 else 0
        deployment_rate = success_rate  # Assuming deployment rate equals success rate
        if avg_build_time is not None and avg_build_time > 0:
            avg_build_time_minutes = round(avg_build_time / 60, 1)
        else:
            avg_build_time_minutes = 0       
        # Calculate MTTR from resolution time estimates
        try:
            mttr_hours = calculate_mttr_for_tool_project(tool, project)
        except Exception as e:
            logger.error(f"Error calculating MTTR for project {project}: {e}")
            mttr_hours =0.0
        
        # Build trend data
        trend_data = []
        for bucket in aggs.get('recent_builds', {}).get('buckets', []):
            trend_data.append({
                "date": bucket['key_as_string'],
                "builds": bucket['doc_count'],
                "success_rate": bucket.get('success_rate', {}).get('value', 0)
            })
        
        metrics = {
            "success_rate": round(success_rate, 1),
            "failure_rate": round(failure_rate, 1),
            "avg_build_time_minutes": round(avg_build_time / 60, 1) if avg_build_time > 0 else 0,
            "deployment_rate": round(deployment_rate, 1),
            "mttr_hours": mttr_hours,
            "total_builds": int(total_builds),
            "successful_builds": int(successful_builds),
            "failed_builds": int(failed_builds),
            "total_errors": int(total_errors),
            "trend_data": trend_data[-7:],  # Last 7 days
            "health_score": calculate_health_score(success_rate, mttr_hours, failure_rate)
        }
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error fetching project metrics: {e}")
        return jsonify({"error": str(e)}), 500

def calculate_mttr_for_tool_project(tool, project):
    """Calculate Mean Time To Recovery for a project"""
    try:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tool": tool}},
                        {"term": {"project": project}},
                        {"term": {"status": "error"}}
                    ]
                }
            },
            "size": 1000,
            "_source": ["resolution_time_estimate"]
        }
        
        response = es.search(index=analysis_index, body=body)
        
        total_resolution_time_hours = 0
        incident_count = 0
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            resolution_estimate = source.get('resolution_time_estimate', '30 minutes')
            
            # Parse resolution time
            if 'hour' in resolution_estimate.lower():
                if '-' in resolution_estimate:
                    hours_range = resolution_estimate.lower().replace('hours', '').replace('hour', '').strip()
                    if '-' in hours_range:
                        start, end = hours_range.split('-')
                        avg_hours = (float(start.strip()) + float(end.strip())) / 2
                    else:
                        avg_hours = float(hours_range)
                else:
                    avg_hours = float(resolution_estimate.lower().replace('hours', '').replace('hour', '').strip())
            elif 'minute' in resolution_estimate.lower():
                minutes_str = resolution_estimate.lower().replace('minutes', '').replace('minute', '').strip()
                minutes = float(re.findall(r'\d+', minutes_str)[0]) if re.findall(r'\d+', minutes_str) else 30
                avg_hours = minutes / 60
            else:
                avg_hours = 1.0
            
            total_resolution_time_hours += avg_hours
            incident_count += 1
        
        return round(total_resolution_time_hours / incident_count, 2) if incident_count > 0 else 0.0
    except Exception as e:
        logger.error(f"Error calculating MTTR: {e}")
        return 0.0

def calculate_health_score(success_rate, mttr_hours, failure_rate):
    """Calculate overall health score for a project"""
    # Weight factors
    success_weight = 0.4
    mttr_weight = 0.3
    failure_weight = 0.3
    
    # Normalize MTTR (lower is better, max expected is 24 hours)
    mttr_score = max(0, 100 - (mttr_hours / 24) * 100)
    failure_score = max(0, 100 - failure_rate)
    
    health_score = (success_rate * success_weight + 
                   mttr_score * mttr_weight + 
                   failure_score * failure_weight)
    
    return round(health_score, 1)

@app.route('/project-analyses/<project>', methods=['GET'])
def get_project_analyses(tool, project):
    """Get real-time error analysis for a project"""
    try:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tool": tool}},
                        {"term": {"project": project}}
                    ]
                }
            },
            "size": 50,
            "sort": [{"analysis_timestamp": {"order": "desc"}}],
            "_source": [
                "failure_category", "severity_level", "business_impact_score",
                "confidence_score", "analysis_timestamp", "environment",
                "server", "error_count", "full_synthesis", "status",
                "affected_components", "resolution_time_estimate"
            ]
        }
        
        response = es.search(index=analysis_index, body=body)
        analyses = []
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Parse full_synthesis if it exists
            synthesis_data = parse_full_synthesis(source.get('full_synthesis'))
            
            analysis_card = {
                "id": hit['_id'],
                "failure_category": source.get('failure_category', 'Unknown'),
                "severity_level": source.get('severity_level', 'medium'),
                "business_impact_score": source.get('business_impact_score', 0.5),
                "confidence_score": source.get('confidence_score', 0.8),
                "timestamp": source.get('analysis_timestamp'),
                "environment": source.get('environment', 'unknown'),
                "server": source.get('server', 'unknown'),
                "error_count": source.get('error_count', 1),
                "status": source.get('status', 'error'),
                "affected_components": source.get('affected_components', []),
                "resolution_time_estimate": source.get('resolution_time_estimate', 'Unknown'),
                "summary": synthesis_data.get('failure_summary', 'No summary available'),
                "root_cause": synthesis_data.get('root_cause', {}),
                "fix_suggestion": synthesis_data.get('fix_suggestion', {}),
                "auto_fix_status": synthesis_data.get('auto_fix', {}).get('status', 'Unknown')
            }
            
            analyses.append(analysis_card)
        
        return jsonify(analyses)
    except Exception as e:
        logger.error(f"Error fetching project analyses: {e}")
        return jsonify({"error": str(e)}), 200

@app.route('/environments/<tool>/<project>', methods=['GET'])
def get_environments_for_project(tool, project):
    """Get all available environments"""
    try:
        body = {
            "query":{
                "bool": {
                    "must": [
                        {"term": {"tool": tool}},
                        {"term": {"project": project}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "environments": {
                    "terms": {"field": "environment", "size": 20}
                }
            }
        }
        
        response = es.search(index=analysis_index, body=body)
        environments = []
        
        for bucket in response['aggregations']['environments']['buckets']:
            environments.append({
                "name": bucket['key'],
                "count": bucket['doc_count']
            })
        
        return jsonify(environments)
    except Exception as e:
        logger.error(f"Error fetching environments: {e}")
        return jsonify({"error": str(e)}), 200

@app.route('/servers/<tool>/<project>/<environment>', methods=['GET'])
def get_servers_for_project(tool, project, environment):
    """Get servers for a specific environment"""
    try:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tool": tool}},
                        {"term": {"project": project}},
                        {"term": {"environment": environment}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "servers": {
                    "terms": {"field": "server", "size": 100},
                    "aggs": {
                        "health": {
                            "terms": {"field": "status"}
                        },
                        "last_seen": {
                            "max": {"field": "analysis_timestamp"}
                        }
                    }
                }
            }
        }
        
        response = es.search(index=analysis_index, body=body)
        servers = []
        
        for bucket in response['aggregations']['servers']['buckets']:
            error_count = 0
            for health_bucket in bucket['health']['buckets']:
                if health_bucket['key'] == 'error':
                    error_count = health_bucket['doc_count']
            
            health_score = max(0, 100 - (error_count / bucket['doc_count']) * 100)
            
            servers.append({
                "name": bucket['key'],
                "total_logs": bucket['doc_count'],
                "error_count": error_count,
                "health_score": round(health_score, 1),
                "last_seen": bucket['last_seen']['value_as_string'],
                "status": "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical"
            })
        
        return jsonify(servers)
    except Exception as e:
        logger.error(f"Error fetching servers: {e}")
        return jsonify({"error": str(e)}), 200

@app.route('/logs/<environment>/<server>', methods=['GET'])
def get_logs(environment, server):
    """Get logs for a specific environment and server"""
    try:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"environment": environment}},
                        {"term": {"server": server}}
                    ]
                }
            },
            "size": 100,
            "sort": [{"analysis_timestamp": {"order": "desc"}}]
        }
        
        response = es.search(index=analysis_index, body=body)
        logs = [hit['_source'] for hit in response['hits']['hits']]
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/pipeline-stages/<tool>/<project>', methods=['GET'])
@app.route('/pipeline-stages/<tool>/<project>/<environment>', methods=['GET'])
@app.route('/pipeline-stages/<tool>/<project>/<environment>/<server>', methods=['GET'])
def get_pipeline_stages(tool, project, environment=None, server=None):
    """Get pipeline stage analysis grouped by log_type"""
    try:
        # Build query filters
        must_filters = [
            {"term": {"tool": tool}},
            {"term": {"project": project}}
        ]
        
        if environment:
            must_filters.append({"term": {"environment": environment}})
        if server:
            must_filters.append({"term": {"server": server}})
        
        body = {
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "size": 1000,
            "sort": [{"analysis_timestamp": {"order": "desc"}}],
            "_source": [
                "log_type", "llm_response", "analysis_timestamp", 
                "status", "severity_level", "confidence_score"
            ]
        }
        
        response = es.search(index=analysis_index, body=body)
        
        # Group by log_type (pipeline stages)
        stages = {
            "git-checkout": [],
            "build": [],
            "test": [],
            "sonarqube-issues": []
        }
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            log_type = source.get('log_type', 'unknown')
            
            if log_type in stages:
                # Parse LLM response
                llm_analysis = parse_llm_response(source.get('llm_response'))
                
                stage_analysis = {
                    "id": hit['_id'],
                    "timestamp": source.get('analysis_timestamp'),
                    "status": source.get('status', 'unknown'),
                    "severity_level": source.get('severity_level', 'medium'),
                    "confidence_score": source.get('confidence_score', 0.8),
                    "analysis": llm_analysis
                }
                
                stages[log_type].append(stage_analysis)
        
        return jsonify(stages)
        
    except Exception as e:
        logger.error(f"Error fetching pipeline stages: {e}")
        return jsonify({
            "git-checkout": [],
            "build": [],
            "test": [],
            "sonarqube-issues": []
        }), 200

def parse_llm_response(llm_response):
    """Parse LLM response JSON string"""
    if llm_response:
        try:
            if isinstance(llm_response, dict):
                return llm_response
            elif isinstance(llm_response, str):
                return json.loads(llm_response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
    
    return {
        "failure_summary": "No analysis available",
        "root_cause": {},
        "fix_suggestion": {},
        "rollback_plan": {},
        "auto_fix": {},
        "severity_level": "unknown",
        "confidence_score": 0.0
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
