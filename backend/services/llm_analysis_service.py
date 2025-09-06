# services/llm_analysis_service.py - CoT analysis for ALL input types
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import json
import re
import requests
from datetime import datetime

app = Flask(__name__)

class UniversalCoTAnalysisService:
    def __init__(self, model_name="llama3.1:8b"):
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_service_url = "http://localhost:5001"
    
    def get_universal_cot_prompt(self, content, metadata):
        """Universal CoT prompt that handles ANY input type"""
        return f"""You are an expert DevOps engineer. Use systematic Chain-of-Thought reasoning to analyze this content regardless of its format (JSON, XML, plain text, logs, YAML, etc.).

CONTENT METADATA:
- Tool: {metadata.get('tool', 'unknown')}
- Project: {metadata.get('project', 'unknown')}
- Environment: {metadata.get('environment', 'unknown')}
- Log Type: {metadata.get('log_type', 'unknown')}
- Status: {metadata.get('status', 'unknown')}

CONTENT TO ANALYZE (ANY FORMAT):
{content}

UNIVERSAL CHAIN-OF-THOUGHT ANALYSIS:
Apply systematic reasoning regardless of input format:

**STEP 1: CONTENT FORMAT IDENTIFICATION**
First, let me identify what type of content this is:
- Is this structured (JSON/XML/YAML) or unstructured (plain text/logs)?
- What format patterns do I see?
- What is the primary information being conveyed?

**STEP 2: CONTENT PARSING AND UNDERSTANDING**
Next, extract the key information:
- What operation or process is described?
- What are the main data points or events?
- What timeline or sequence is present?

**STEP 3: ISSUE DETECTION AND CLASSIFICATION**
Now scan for problems systematically:
- Are there explicit error messages or failure indicators?
- What warning signs or anomalies exist?
- What success indicators are present?
- How severe are any issues found?

**STEP 4: CONTEXTUAL ANALYSIS**
Consider the operational context:
- How does this relate to {metadata.get('tool', 'unknown')} operations?
- What is the significance in {metadata.get('environment', 'unknown')} environment?
- What are the business implications?

**STEP 5: ROOT CAUSE REASONING**
Determine underlying causes:
- What is the primary technical cause of any issues?
- What contributing factors exist?
- How confident am I in this assessment?

**STEP 6: SOLUTION DEVELOPMENT**
Formulate actionable solutions:
- What specific steps will resolve identified issues?
- What commands or configurations are needed?
- What verification steps should follow?

**STEP 7: PREVENTION STRATEGY**
Plan for future prevention:
- How can similar issues be prevented?
- What monitoring should be implemented?
- What process improvements are needed?

Based on this systematic Chain-of-Thought analysis, provide comprehensive insights about this content.

RESPONSE FORMAT: Provide detailed analysis in natural language covering all the reasoning steps above. Be specific about findings, solutions, and recommendations.
"""
    
    def analyze_with_universal_cot(self, content, metadata):
        """Universal CoT analysis that works with ANY input format"""
        try:
            start_time = datetime.now()
            
            print(f"🧠 Starting universal CoT analysis for {metadata.get('tool', 'unknown')} {metadata.get('log_type', 'unknown')} (any format)")
            
            # Generate universal CoT prompt
            prompt = self.get_universal_cot_prompt(content, metadata)
            
            # Get LLM response with CoT reasoning
            llm_response = self.llm.invoke(prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            print(f"📝 Received universal CoT response ({len(llm_response)} chars) in {processing_time:.0f}ms")
            
            # Extract structured insights using CoT principles
            structured_analysis = self.extract_structured_insights_with_cot(llm_response, content, metadata)
            structured_analysis['processing_time_ms'] = processing_time
            structured_analysis['llm_response'] = llm_response
            structured_analysis['analysis_method'] = 'Universal Chain-of-Thought reasoning'
            
            print(f"✅ Universal CoT analysis completed for {metadata.get('tool', 'unknown')} {metadata.get('log_type', 'unknown')}")
            return structured_analysis
            
        except Exception as e:
            print(f"❌ Universal CoT analysis failed: {e}")
            return self.create_cot_emergency_analysis(metadata, str(e))
    
    def extract_structured_insights_with_cot(self, llm_response, original_content, metadata):
        """Extract structured insights using CoT reasoning from any response format"""
        
        # Use LLM to extract structured data with CoT
        extraction_prompt = f"""Using Chain-of-Thought reasoning, extract structured information from this analysis:

ORIGINAL ANALYSIS:
{llm_response}

ORIGINAL CONTENT CONTEXT:
{original_content[:500]}...

METADATA:
- Tool: {metadata.get('tool', 'unknown')}
- Environment: {metadata.get('environment', 'unknown')}
- Log Type: {metadata.get('log_type', 'unknown')}

CHAIN-OF-THOUGHT EXTRACTION:

Step 1: Identify the main findings from the analysis
Step 2: Extract specific technical details and metrics
Step 3: Determine severity and impact levels
Step 4: Extract actionable recommendations
Step 5: Assess confidence and complexity

Based on this reasoning, provide the following information:

EXECUTIVE_SUMMARY: [brief summary of findings]
ROOT_CAUSE: [primary technical cause with confidence %]
FIX_STRATEGY: [specific actionable steps]
ROLLBACK_PLAN: [safe rollback procedures]
AUTO_FIX_FEASIBILITY: [can this be automated safely?]
SEVERITY_LEVEL: [low/medium/high/critical]
CONFIDENCE_SCORE: [0.0-1.0]
ERROR_COUNT: [number of errors found]
WARNING_COUNT: [number of warnings found]
SUCCESS_INDICATORS: [number of success markers]
RESOLUTION_TIME: [estimated time to resolve]
BUSINESS_IMPACT: [0.0-1.0 scale]
TECHNICAL_COMPLEXITY: [low/medium/high]
MONITORING_RECOMMENDATIONS: [specific monitoring advice]
"""
        
        try:
            structured_response = self.llm.invoke(extraction_prompt)
            return self.parse_cot_structured_response(structured_response, llm_response, original_content, metadata)
        except Exception as e:
            print(f"⚠️ CoT structured extraction failed: {e}")
            return self.pattern_based_cot_analysis(llm_response, original_content, metadata)
    
    def parse_cot_structured_response(self, structured_response, original_response, original_content, metadata):
        """Parse CoT structured response into final analysis"""
        
        def extract_cot_field(pattern, default=""):
            match = re.search(f"{pattern}:\s*(.+?)(?=\n[A-Z_]+:|$)", structured_response, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else default
        
        def extract_cot_numeric(pattern, default=0):
            try:
                text = extract_cot_field(pattern, str(default))
                numbers = re.findall(r'\d+\.?\d*', text)
                return float(numbers[0]) if numbers else default
            except:
                return default
        
        # Extract auto-fix feasibility
        auto_fix_text = extract_cot_field("AUTO_FIX_FEASIBILITY", "Manual intervention required")
        auto_fix = "Manual intervention required"
        if any(word in auto_fix_text.lower() for word in ["yes", "safe", "automated", "can be automated"]):
            auto_fix = f"Automated fix possible: {auto_fix_text[:100]}"
        
        return {
            "failure_summary": extract_cot_field("EXECUTIVE_SUMMARY", f"CoT analysis of {metadata.get('tool', 'unknown')} {metadata.get('log_type', 'unknown')} content"),
            "root_cause": extract_cot_field("ROOT_CAUSE", "CoT analysis identified technical factors"),
            "fix_suggestion": extract_cot_field("FIX_STRATEGY", f"Apply {metadata.get('tool', 'unknown')} best practices"),
            "rollback_plan": extract_cot_field("ROLLBACK_PLAN", f"Standard {metadata.get('tool', 'unknown')} rollback procedures"),
            "auto_fix": auto_fix,
            "severity_level": extract_cot_field("SEVERITY_LEVEL", "medium").lower(),
            "confidence_score": min(max(extract_cot_numeric("CONFIDENCE_SCORE", 0.8), 0.0), 1.0),
            "error_count": int(extract_cot_numeric("ERROR_COUNT", 0)),
            "warning_count": int(extract_cot_numeric("WARNING_COUNT", 0)),
            "success_indicators": int(extract_cot_numeric("SUCCESS_INDICATORS", 0)),
            "resolution_time_estimate": extract_cot_field("RESOLUTION_TIME", "2-4 hours"),
            "business_impact_score": min(max(extract_cot_numeric("BUSINESS_IMPACT", 0.5), 0.0), 1.0),
            "technical_complexity": extract_cot_field("TECHNICAL_COMPLEXITY", "medium").lower(),
            "failure_category": f"{metadata.get('log_type', 'general')}_cot_analysis",
            "affected_components": [metadata.get('tool', 'unknown')],
            "build_duration_seconds": 0,
            "test_pass_rate": 0.0,
            "deployment_success": metadata.get('status') == 'success',
            "monitoring_recommendations": extract_cot_field("MONITORING_RECOMMENDATIONS", f"Monitor {metadata.get('tool', 'unknown')} operations"),
            "full_synthesis": original_response,
            "cot_reasoning_applied": True,
            "input_format": "universal",
            "processing_method": "universal_cot"
        }
    
    def pattern_based_cot_analysis(self, llm_response, original_content, metadata):
        """Pattern-based CoT analysis when structured extraction fails"""
        
        print("🔄 Applying pattern-based CoT analysis...")
        
        # CoT Step 1: Content analysis
        error_count = len(re.findall(r'(?:error|Error|ERROR|failed|Failed|FAILED)', original_content, re.IGNORECASE))
        warning_count = len(re.findall(r'(?:warning|Warning|WARNING|warn|Warn)', original_content, re.IGNORECASE))
        success_indicators = len(re.findall(r'(?:success|Success|SUCCESS|passed|completed|ok)', original_content, re.IGNORECASE))
        
        # CoT Step 2: Severity assessment
        severity = "low"
        if re.search(r'\b(?:critical|fatal|severe|emergency)\b', original_content, re.IGNORECASE):
            severity = "critical"
        elif re.search(r'\b(?:high|major|important|urgent|error|failed)\b', original_content, re.IGNORECASE):
            severity = "high"
        elif re.search(r'\b(?:medium|moderate|warning)\b', original_content, re.IGNORECASE):
            severity = "medium"
        
        # CoT Step 3: Business impact reasoning
        business_impact = 0.3
        if severity == "critical":
            business_impact = 0.9
        elif severity == "high":
            business_impact = 0.7
        elif severity == "medium":
            business_impact = 0.5
        elif metadata.get('status') == 'success':
            business_impact = 0.1
        
        # CoT Step 4: Extract key insights
        def extract_key_insight(text, max_length=200):
            sentences = re.split(r'[.!?]+', text)
            relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return '. '.join(relevant_sentences[:2])[:max_length] if relevant_sentences else "CoT analysis applied"
        
        return {
            "failure_summary": f"CoT Pattern Analysis: {extract_key_insight(llm_response, 250)}",
            "root_cause": f"CoT Investigation: Pattern-based analysis of {metadata.get('tool', 'unknown')} {metadata.get('log_type', 'unknown')} content (Confidence: 70%)",
            "fix_suggestion": f"CoT Solution: Apply systematic troubleshooting for {metadata.get('tool', 'unknown')} in {metadata.get('environment', 'unknown')} environment",
            "rollback_plan": f"CoT Rollback: Standard {metadata.get('tool', 'unknown')} rollback procedures with verification",
            "auto_fix": "Manual intervention required - CoT analysis recommends careful review",
            "severity_level": severity,
            "confidence_score": 0.7,
            "error_count": error_count,
            "warning_count": warning_count,
            "success_indicators": success_indicators,
            "resolution_time_estimate": "2-4 hours" if severity in ["high", "critical"] else "1-2 days",
            "business_impact_score": business_impact,
            "technical_complexity": "medium",
            "failure_category": f"{metadata.get('log_type', 'general')}_pattern_cot",
            "affected_components": [metadata.get('tool', 'unknown')],
            "build_duration_seconds": 0,
            "test_pass_rate": 0.0,
            "deployment_success": metadata.get('status') == 'success',
            "monitoring_recommendations": f"CoT Prevention: Systematic monitoring for {metadata.get('tool', 'unknown')} operations",
            "full_synthesis": llm_response,
            "cot_reasoning_applied": True,
            "input_format": "universal",
            "processing_method": "pattern_based_cot"
        }
    
    def create_cot_emergency_analysis(self, metadata, error_msg):
        """Emergency CoT analysis when all methods fail"""
        return {
            "failure_summary": f"Emergency CoT Analysis: System error in processing {metadata.get('tool', 'unknown')} {metadata.get('log_type', 'unknown')} content",
            "root_cause": f"CoT Emergency Protocol: Analysis system encountered error - {error_msg[:100]} (Confidence: 30%)",
            "fix_suggestion": f"CoT Emergency Response: Manual review required for {metadata.get('tool', 'unknown')} content",
            "rollback_plan": f"Emergency rollback: Standard {metadata.get('tool', 'unknown')} procedures",
            "auto_fix": "Manual intervention required - CoT emergency protocol",
            "severity_level": "medium",
            "confidence_score": 0.3,
            "error_count": 0,
            "warning_count": 1,
            "success_indicators": 0,
            "resolution_time_estimate": "immediate",
            "business_impact_score": 0.4,
            "technical_complexity": "low",
            "failure_category": "cot_emergency_analysis",
            "affected_components": ["cot_analysis_service"],
            "build_duration_seconds": 0,
            "test_pass_rate": 0.0,
            "deployment_success": False,
            "monitoring_recommendations": "Monitor CoT analysis service health",
            "full_synthesis": f"Emergency CoT processing due to: {error_msg}",
            "cot_reasoning_applied": True,
            "input_format": "emergency",
            "processing_method": "emergency_cot",
            "system_error": error_msg
        }
    
    def store_in_both_tables(self, analysis_result, metadata):
        """Store CoT analysis in both cicd_analysis and cicd_vectors tables"""
        
        print("💾 Storing universal CoT analysis in both tables...")
        
        # Store in cicd_analysis table
        analysis_success = self.store_analysis_table(analysis_result, metadata)
        
        # Store in cicd_vectors table for RAG
        vector_success = self.store_vector_table(analysis_result, metadata)
        
        if analysis_success and vector_success:
            print("✅ Universal CoT analysis stored in both tables")
        else:
            print("⚠️ Partial storage success")
        
        return analysis_success and vector_success
    
    def store_analysis_table(self, analysis_result, metadata):
        """Store in cicd_analysis table"""
        try:
            analysis_data = {
                "log_id": metadata.get('log_id'),
                "correlation_id": metadata.get('correlation_id'),
                "tool": metadata.get('tool'),
                "project": metadata.get('project'),
                "environment": metadata.get('environment'),
                "server": metadata.get('server'),
                "log_type": metadata.get('log_type'),
                "status": metadata.get('status'),
                "llm_model": "llama3.1:8b",
                "analysis": analysis_result
            }
            
            response = requests.post(f"{self.db_service_url}/store-analysis", json=analysis_data, timeout=60)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error storing in cicd_analysis: {e}")
            return False
    
    def store_vector_table(self, analysis_result, metadata):
        """Store in cicd_vectors table for RAG"""
        try:
            # Create content for embedding
            content_parts = [
                analysis_result.get('failure_summary', ''),
                analysis_result.get('root_cause', ''),
                analysis_result.get('fix_suggestion', ''),
                analysis_result.get('monitoring_recommendations', '')
            ]
            
            content = ' '.join(filter(None, content_parts))
            
            if content:
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                
                # Store vector
                vector_data = {
                    "content": content,
                    "content_vector": embedding,
                    "correlation_id": metadata.get('correlation_id'),
                    "tool": metadata.get('tool'),
                    "project": metadata.get('project'),
                    "environment": metadata.get('environment'),
                    "log_type": metadata.get('log_type'),
                    "status": metadata.get('status', 'unknown'),
                    "error_pattern": f"{metadata.get('tool')}_{metadata.get('log_type')}_{analysis_result.get('failure_category')}",
                    "solution_type": analysis_result.get('severity_level'),
                    "tags": [
                        metadata.get('tool'), 
                        metadata.get('environment'), 
                        metadata.get('log_type'),
                        metadata.get('status'),
                        analysis_result.get('severity_level'),
                        'universal_cot_analysis'
                    ]
                }
                
                response = requests.post(f"{self.db_service_url}/store-vector", json=vector_data, timeout=30)
                return response.status_code == 200
            
            return False
        except Exception as e:
            print(f"❌ Error storing in cicd_vectors: {e}")
            return False

# Initialize service
llm_service = UniversalCoTAnalysisService()

@app.route('/analyze', methods=['POST'])
def analyze_universal_content():
    """Universal CoT analysis endpoint for ANY input type"""
    try:
        # Handle ANY input format
        content = ""
        metadata = {}
        
        if request.is_json:
            try:
                data = request.get_json()
                content = data.get('content', '')
                metadata = data.get('metadata', {})
            except:
                # If JSON parsing fails, treat as raw content
                content = request.get_data(as_text=True)
        else:
            # Handle raw text, XML, YAML, logs, etc.
            content = request.get_data(as_text=True)
        
        # Validate content
        if not content or len(content.strip()) < 3:
            return jsonify({
                'status': 'skipped',
                'reason': 'Insufficient content for CoT analysis'
            }), 200
        
        print(f"🔍 Received universal content for CoT analysis ({len(content)} chars)")
        
        # Analyze with universal CoT reasoning
        analysis_result = llm_service.analyze_with_universal_cot(content, metadata)
        
        # Store in both tables
        storage_success = llm_service.store_in_both_tables(analysis_result, metadata)
        
        # Return results
        return jsonify({
            'status': 'success',
            'results': {
                'final_analysis': analysis_result,
                'chunk_results': [analysis_result],
                'processing_stats': {
                    'chunks_processed': 1,
                    'processing_time': datetime.utcnow().isoformat(),
                    'model_used': 'llama3.1:8b',
                    'analysis_method': 'Universal Chain-of-Thought reasoning',
                    'input_format': 'any_type',
                    'cot_reasoning_applied': True,
                    'storage_success': storage_success,
                    'tables_updated': ['cicd_analysis', 'cicd_vectors']
                }
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Universal CoT analysis error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'cot_emergency_applied': True
        }), 200  # Return 200 to continue pipeline

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        test_response = llm_service.llm.invoke("Hello")
        llm_healthy = len(test_response) > 0
        
        return jsonify({
            'status': 'healthy',
            'service': 'Universal CoT LLM Analysis Service',
            'model': 'llama3.1:8b',
            'analysis_method': 'Universal Chain-of-Thought reasoning',
            'input_support': 'ALL formats (JSON, XML, YAML, text, logs, etc.)',
            'cot_steps': 7,
            'storage_targets': ['cicd_analysis', 'cicd_vectors'],
            'llm_healthy': llm_healthy
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    print("🧠 Starting Universal Chain-of-Thought LLM Analysis Service")
    print("🦙 Model: Llama 3.1 8B with universal CoT reasoning")
    print("📥 Input: ANY format (JSON, XML, YAML, text, logs, binary, etc.)")
    print("🔧 CoT Steps: Format ID → Parsing → Issue Detection → Context → Root Cause → Solution → Prevention")
    print("📊 Storage: Both cicd_analysis and cicd_vectors tables")
    print("⚡ Features: Universal input handling, systematic reasoning, dual storage")
    app.run(debug=False, port=5003, use_reloader=False)
