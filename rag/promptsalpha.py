import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import AzureChatOpenAI
import os

def incident_to_content(incident):
    return (
        f"EXECUTIVE_SUMMARY: {incident.get('executive_summary', '')}\n"
        f"ROOT_CAUSE: {incident.get('root_cause', '')}\n"
        f"FIX_STRATEGY: {incident.get('fix_strategy', '')}\n"
        f"ROLLBACK_PLAN: {incident.get('rollback_plan', '')}\n"
        f"AUTO_FIX_FEASIBILITY: {incident.get('auto_fix_feasibility', '')}\n"
        f"SEVERITY_LEVEL: {incident.get('severity_level', '')}\n"
        f"CONFIDENCE_SCORE: {incident.get('confidence_score', '')}\n"
        f"ERROR_COUNT: {incident.get('error_count', '')}\n"
        f"WARNING_COUNT: {incident.get('warning_count', '')}\n"
        f"SUCCESS_INDICATORS: {incident.get('success_indicators', '')}\n"
        f"RESOLUTION_TIME: {incident.get('resolution_time', '')}\n"
        f"BUSINESS_IMPACT: {incident.get('business_impact', '')}\n"
        f"TECHNICAL_COMPLEXITY: {incident.get('technical_complexity', '')}\n"
        f"MONITORING_RECOMMENDATIONS: {incident.get('monitoring_recommendations', '')}"
    )


prompts = [
    f"""
You are an expert incident analyst. Carefully review the provided technical data or incident log and follow these steps:
1. Content Identification:
   - Specify the type of content (e.g., system log, incident report, monitoring alert) and its main objective.
2. Key Findings Extraction:
   - List all major events, errors, warnings, and any indicators of successful operation.
3. Root Cause Analysis:
   - Diagnose the primary technical root cause(s) of any issues, including supporting evidence and your confidence percentage.
4. Severity & Impact Assessment:
   - Assess the severity (low/medium/high/critical) and estimate the business impact on a 0.0–1.0 scale.
5. Actionable Recommendations:
   - Provide specific, prioritized fixes and preventive actions.
   - Suggest monitoring improvements and safe rollback procedures.
   - Evaluate if an automated fix is feasible.

Respond using this structured format:
EXECUTIVE_SUMMARY: [Concise overview of findings]
ROOT_CAUSE: [Main technical cause(s) with confidence %]
FIX_STRATEGY: [Clear, actionable remediation steps]
ROLLBACK_PLAN: [Safe rollback steps, if applicable]
AUTO_FIX_FEASIBILITY: [Yes/No, with justification]
SEVERITY_LEVEL: [low/medium/high/critical]
CONFIDENCE_SCORE: [0.0–1.0]
ERROR_COUNT: [Number of errors]
WARNING_COUNT: [Number of warnings]
SUCCESS_INDICATORS: [Number of success markers]
RESOLUTION_TIME: [Estimated time to resolve]
BUSINESS_IMPACT: [0.0–1.0]
TECHNICAL_COMPLEXITY: [low/medium/high]
MONITORING_RECOMMENDATIONS: [Specific advice]

Be concise, factual, and avoid unnecessary commentary.
""",


    f"""
Analyze the following CI/CD pipeline incident. For each section, provide clear, evidence-based details:

EXECUTIVE_SUMMARY: [One-sentence summary]
ROOT_CAUSE: [Describe the main technical cause, include confidence % and reasoning]
FIX_STRATEGY: [List actionable remediation steps]
ROLLBACK_PLAN: [Describe rollback process, if applicable]
AUTO_FIX_FEASIBILITY: [Yes/No, with rationale]
SEVERITY_LEVEL: [low/medium/high/critical]
CONFIDENCE_SCORE: [0.0–1.0]
ERROR_COUNT: [#]
WARNING_COUNT: [#]
SUCCESS_INDICATORS: [#]
RESOLUTION_TIME: [Estimated time]
BUSINESS_IMPACT: [0.0–1.0]
TECHNICAL_COMPLEXITY: [low/medium/high]
MONITORING_RECOMMENDATIONS: [Advice for future prevention]

Stick to the format and use information from the incident log only.

""",
f"""
For the following CI/CD incident, provide a structured analysis using the exact fields below. Base your answers strictly on the incident details:

EXECUTIVE_SUMMARY: [Incident summary]
ROOT_CAUSE: [Technical cause(s), confidence %, and log evidence]
FIX_STRATEGY: [Remediation steps]
ROLLBACK_PLAN: [Rollback plan, if needed]
AUTO_FIX_FEASIBILITY: [Yes/No, with reason]
SEVERITY_LEVEL: [low/medium/high/critical]
CONFIDENCE_SCORE: [0.0–1.0]
ERROR_COUNT: [Count]
WARNING_COUNT: [Count]
SUCCESS_INDICATORS: [Count]
RESOLUTION_TIME: [Estimated time]
BUSINESS_IMPACT: [0.0–1.0]
TECHNICAL_COMPLEXITY: [low/medium/high]
MONITORING_RECOMMENDATIONS: [Monitoring suggestions]

Do not add extra commentary or fields.


""",

f"""
Given the incident data below, fill out each field with specific and factual information:

EXECUTIVE_SUMMARY: [Concise overview of the incident]
ROOT_CAUSE: [What caused the issue, with confidence % and supporting log evidence]
FIX_STRATEGY: [How to resolve the issue]
ROLLBACK_PLAN: [Steps to revert if needed]
AUTO_FIX_FEASIBILITY: [Yes/No, justify]
SEVERITY_LEVEL: [low/medium/high/critical]
CONFIDENCE_SCORE: [0.0–1.0]
ERROR_COUNT: [Total errors]
WARNING_COUNT: [Total warnings]
SUCCESS_INDICATORS: [Success signals]
RESOLUTION_TIME: [Estimated fix time]
BUSINESS_IMPACT: [0.0–1.0]
TECHNICAL_COMPLEXITY: [low/medium/high]
MONITORING_RECOMMENDATIONS: [How to monitor/prevent recurrence]

Do not include information not present in the log.


""",

f""" 
You are an expert incident analyst. Analyze the given technical data or incident log and follow these steps:

1. Identify Content:
   - Type (e.g., system log, incident report, alert)
   - Primary purpose

2. Extract Key Findings:
   - List major events, errors, warnings, and successful operations

3. Analyze Root Cause:
   - Determine the technical root cause(s) with evidence and confidence %

4. Assess Severity & Impact:
   - Severity (low/medium/high/critical)
   - Business impact (0.0–1.0)

5. Recommend Actions:
   - Prioritized fixes and preventive steps
   - Monitoring improvements
   - Rollback steps, if needed
   - Automation feasibility

Use this response format:

EXECUTIVE_SUMMARY: [Brief overview of findings]  
ROOT_CAUSE: [Primary technical issue(s) with confidence %]  
FIX_STRATEGY: [Actionable remediation steps]  
ROLLBACK_PLAN: [Steps for safe rollback]  
AUTO_FIX_FEASIBILITY: [Yes/No, with reasoning]  
SEVERITY_LEVEL: [low/medium/high/critical]  
CONFIDENCE_SCORE: [0.0–1.0]  
ERROR_COUNT: [Number of errors]  
WARNING_COUNT: [Number of warnings]  
SUCCESS_INDICATORS: [Number of success events]  
RESOLUTION_TIME: [Estimated time to resolve]  
BUSINESS_IMPACT: [0.0–1.0]  
TECHNICAL_COMPLEXITY: [low/medium/high]  
MONITORING_RECOMMENDATIONS: [Specific monitoring suggestions]

Be precise, objective, and avoid filler commentary.
"""
]


# 3. Define ground truth
GROUND_TRUTH = [

  {
    "incident_id": 1,
    "executive_summary": "Authentication failure with Docker registry caused build and deployment to fail",
    "root_cause": "Incorrect or expired Docker registry credentials leading to authentication failure (Confidence: 0.9)",
    "fix_strategy": "1. Verify and update Docker registry credentials\n2. Test authentication manually before automated builds\n3. Implement credential rotation and secure storage\n4. Add pre-deployment credential validation step",
    "rollback_plan": "Revert to last successful build and deployment using previously validated Docker image",
    "auto_fix_feasibility": "No",
    "severity_level": "High",
    "confidence_score": 0.9,
    "error_count": 1,
    "warning_count": 0,
    "success_indicators": 0,
    "resolution_time": "1–2 hours",
    "business_impact": 0.7,
    "technical_complexity": "Low",
    "monitoring_recommendations": "Implement monitoring for failed authentication attempts to Docker registry and alert on repeated failures. Add build pipeline step to validate credentials before deployment"
  },
  {
    "incident_id": 2,
    "executive_summary": "Intermittent failure occurred in the integration test suite, resolved successfully upon retry without further errors",
    "root_cause": "Transient instability or timing-related issue in the integration test environment causing intermittent test failures. Confidence 0.75",
    "fix_strategy": "- Investigate test environment stability and resource availability during test runs\n- Review and improve test suite timing and dependencies to reduce flakiness\n- Implement retries with backoff as a temporary mitigation\n- Enhance test logging to capture failure context for future analysis",
    "rollback_plan": "Not applicable as no deployment or configuration change caused the failure",
    "auto_fix_feasibility": "Yes",
    "severity_level": "Low",
    "confidence_score": 0.75,
    "error_count": 1,
    "warning_count": 1,
    "success_indicators": 1,
    "resolution_time": "~1 minute (time between failure and success on retry)",
    "business_impact": 0.1,
    "technical_complexity": "Low",
    "monitoring_recommendations": "- Implement monitoring on test environment resource utilization and availability\n- Track frequency and patterns of test suite failures to identify trends\n- Alert on repeated test failures beyond retry threshold"
  },
  {
    "incident_id": 3,
    "executive_summary": "Deployment halted due to inability to connect to the production database server db-prod-01, causing a critical failure in the deployment process",
    "root_cause": "Database server db-prod-01 refused connection, likely due to network issues, server downtime, or configuration errors (confidence 0.85)",
    "fix_strategy": "- Verify database server status and network connectivity\n- Check firewall and access control settings for db-prod-01\n- Review recent changes to database or network configurations\n- Restart database service if unresponsive\n- Implement retry logic in deployment scripts for transient failures",
    "rollback_plan": "- Revert deployment to last known good state before current deployment attempt\n- Confirm database connectivity before redeployment",
    "auto_fix_feasibility": "No",
    "severity_level": "critical",
    "confidence_score": 0.85,
    "error_count": 1,
    "warning_count": 0,
    "success_indicators": 0,
    "resolution_time": "1-2 hours (dependent on database recovery)",
    "business_impact": 0.9,
    "technical_complexity": "medium",
    "monitoring_recommendations": "- Implement proactive database availability and connectivity monitoring with alerting\n- Monitor network latency and firewall rule changes affecting db-prod-01\n- Add deployment pre-checks for database connectivity before starting deployment"
  },
  {
    "incident_id": 4,
    "executive_summary": "Build process failed due to an invalid environment variable 'API_KEY', causing termination of the build",
    "root_cause": "Misconfigured or missing 'API_KEY' environment variable leading to build failure (confidence 0.95)",
    "fix_strategy": "Validate and correct the 'API_KEY' environment variable value before build initiation; implement environment variable validation checks in CI pipeline",
    "rollback_plan": "Revert to last successful build configuration and environment variable settings to restore build functionality",
    "auto_fix_feasibility": "Yes",
    "severity_level": "high",
    "confidence_score": 0.95,
    "error_count": 1,
    "warning_count": 0,
    "success_indicators": 0,
    "resolution_time": "1-2 hours",
    "business_impact": 0.7,
    "technical_complexity": "low",
    "monitoring_recommendations": "Implement environment variable validation monitoring in CI/CD pipeline; alert on invalid or missing critical environment variables before build start"
  },
  {
    "incident_id": 5,
    "executive_summary": "A flaky test ('test_user_signup') caused intermittent failures requiring retries, but all tests ultimately passed after two retries",
    "root_cause": "Intermittent instability in the 'test_user_signup' test environment or test code causing flakiness (confidence 0.85)",
    "fix_strategy": "- Investigate and stabilize the 'test_user_signup' test to eliminate flakiness\n- Review test dependencies and environment consistency\n- Implement retries as a temporary mitigation\n- Enhance test logging for better failure diagnostics",
    "rollback_plan": "Not applicable as no deployment or code change caused failure; issue is test instability",
    "auto_fix_feasibility": "No",
    "severity_level": "Low",
    "confidence_score": 0.85,
    "error_count": 0,
    "warning_count": 1,
    "success_indicators": 1,
    "resolution_time": "1-2 days (to identify and fix flaky test)",
    "business_impact": 0.1,
    "technical_complexity": "Medium",
    "monitoring_recommendations": "- Implement test flakiness tracking dashboards\n- Alert on flaky test occurrences and retry counts\n- Monitor test environment stability metrics"
  },
  {
    "incident_id": 6,
    "executive_summary": "Docker build process failed due to inability to fetch the base image caused by network connectivity issues",
    "root_cause": "Network connectivity failure preventing access to the Docker image repository (Confidence: 0.95)",
    "fix_strategy": "1. Verify and restore network connectivity to the Docker registry\n2. Check firewall and proxy settings that may block outbound connections\n3. Validate DNS resolution for the image repository domain\n4. Retry the Docker build after network restoration\n5. Implement network health checks prior to build initiation",
    "rollback_plan": "No rollback needed as the failure occurred during build; previous successful image can be redeployed",
    "auto_fix_feasibility": "Yes",
    "severity_level": "Medium",
    "confidence_score": 0.95,
    "error_count": 1,
    "warning_count": 0,
    "success_indicators": 0,
    "resolution_time": "1–2 hours",
    "business_impact": 0.3,
    "technical_complexity": "Low",
    "monitoring_recommendations": "Implement network availability monitoring for build servers and Docker registry endpoints; alert on connectivity failures and DNS resolution errors"
  }

]
# 4. Azure OpenAI setup
os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_KEY_HERE"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://intern-openai-demo.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name="gpt-4.1-mini",
    temperature=0
)

# 5. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def tfidf_cosine(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf_matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def extract_fields_from_llm_output(llm_output):
    fields = {
        'EXECUTIVE_SUMMARY': '',
        'ROOT_CAUSE': '',
        'FIX_STRATEGY': '',
        'SEVERITY_LEVEL': ''
    }
    current_field = None
    for line in llm_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        for field in fields:
            if line.startswith(field + ":"):
                current_field = field
                fields[field] = line.split(':', 1)[1].strip()
                break
        else:
            if current_field and fields[current_field]:
                fields[current_field] += ' ' + line
    return fields

def actionable_suggestions():
    return [
        "Review clarity and specificity: Ensure the prompt clearly defines the task, expected output format, and any constraints.",
        "Add or improve context: Provide relevant background information or examples to help the model understand the scenario.",
        "Refine structure: Use structured, numbered instructions and explicit field headers for each required output.",
        "Encourage step-by-step reasoning: Ask the model to explain its reasoning or cite log evidence for conclusions.",
        "Balance brevity and detail: Remove unnecessary verbosity, but do not omit essential guidance.",
        "Include explicit evaluation criteria: Indicate how outputs will be judged (e.g., accuracy, completeness, format adherence).",
        "Test with varied incidents: Evaluate prompt on diverse incident types to find weaknesses and edge cases.",
        "Iterate and compare: Make small prompt changes, re-evaluate, and keep the best-performing versions."
    ]

# 6. Main evaluation loop
prompt_scores = []
for prompt_idx, prompt_template in enumerate(prompts):
    all_scores = []
    for incident in GROUND_TRUTH:
        prompt = prompt_template + "\n\nINCIDENT_DATA:\n" + incident_to_content(incident)
        response = llm.invoke(prompt)
        llm_output = response.content
        llm_fields = extract_fields_from_llm_output(llm_output)
        gt_fields = {
            'EXECUTIVE_SUMMARY': incident.get('executive_summary', ''),
            'ROOT_CAUSE': incident['root_cause'],
            'FIX_STRATEGY': incident['fix_strategy'],
            'SEVERITY_LEVEL': incident['severity_level']
        }
        field_scores = []
        for field in gt_fields:
            if gt_fields[field] and llm_fields.get(field):
                tfidf_score = tfidf_cosine(gt_fields[field], llm_fields[field])
                sem_score = semantic_similarity(gt_fields[field], llm_fields[field])
                avg_score = (tfidf_score + sem_score) / 2
                field_scores.append(avg_score)
        if field_scores:
            all_scores.append(np.mean(field_scores))
    avg_prompt_score = np.mean(all_scores) if all_scores else 0
    prompt_scores.append((prompt_idx, avg_prompt_score))


# 7. Output results
print("\nPrompt Evaluation Results for CI/CD Analyzer:\n")
print("{:<10} {:<10}".format("Prompt ID", "Score"))
for idx, score in prompt_scores:
    print("{:<10} {:<10.4f}".format(idx, score))

# Identify best prompt
best_prompt_idx, best_score = max(prompt_scores, key=lambda x: x[1])
print(f"\nBest Prompt: #{best_prompt_idx} (Score: {best_score:.4f})")

# 8. If best score is below 0.8, suggest actionable items
if best_score < 0.8:
    print("\n⚠️  Best score is below 0.8. Suggested Actionable Items to Improve Prompts:\n")
    for i, suggestion in enumerate(actionable_suggestions(), 1):
        print(f"{i}. {suggestion}")

# Save detailed results with UTF-8 encoding
with open('cd_prompt_evaluation.txt', 'w', encoding='utf-8') as f:
    f.write("CI/CD Prompt Evaluation Results\n")
    f.write("="*50 + "\n")
    f.write("{:<10} {:<10}\n".format("Prompt ID", "Score"))
    for idx, score in prompt_scores:
        f.write("{:<10} {:<10.4f}\n".format(idx, score))
    f.write(f"\nBest Prompt: #{best_prompt_idx} (Score: {best_score:.4f})\n")
    if best_score < 0.8:
        f.write("\n⚠️  Best score is below 0.8. Suggested Actionable Items to Improve Prompts:\n")
        for i, suggestion in enumerate(actionable_suggestions(), 1):
            f.write(f"{i}. {suggestion}\n")

