import numpy as np
from sentence_transformers import SentenceTransformer, util

def normalize(text):
    if isinstance(text, list):
        text = ' '.join(str(x) for x in text)
    return ' '.join(str(text).lower().strip().split())

def incident_to_flat_string(incident):
    items = []
    for k in sorted(incident.keys()):
        if k == "incident_id":
            continue
        items.append(f"{k.upper()}: {normalize(incident[k])}")
    return '\n'.join(items)

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

# 3. Define ground truth (ensure incident_id is present)
GROUND_TRUTH = [
    {
        "incident_id": 0,
        "EXECUTIVE_SUMMARY": "Authentication failure with Docker registry caused build and deployment to fail",
        "ROOT_CAUSE": "Incorrect or expired Docker registry credentials leading to authentication failure (Confidence: 0.9)",
        "FIX_STRATEGY": [
            "Verify and update Docker registry credentials",
            "Test authentication manually before automated builds",
            "Implement credential rotation and secure storage",
            "Add pre-deployment credential validation step"
        ],
        "ROLLBACK_PLAN": "Revert to last successful build and deployment using previously validated Docker image",
        "AUTO_FIX_FEASIBILITY": "No",
        "SEVERITY_LEVEL": "High",
        "CONFIDENCE_SCORE": 0.9,
        "ERROR_COUNT": 1,
        "WARNING_COUNT": 0,
        "SUCCESS_INDICATORS": 0,
        "RESOLUTION_TIME": "1–2 hours",
        "BUSINESS_IMPACT": 0.7,
        "TECHNICAL_COMPLEXITY": "Low",
        "MONITORING_RECOMMENDATIONS": "Implement monitoring for failed authentication attempts to Docker registry and alert on repeated failures. Add build pipeline step to validate credentials before deployment"
    },
    {
        "incident_id": 1,
        "EXECUTIVE_SUMMARY": "Intermittent failure occurred in the integration test suite, resolved successfully upon retry without further errors",
        "ROOT_CAUSE": "Transient instability or timing-related issue in the integration test environment causing intermittent test failures. Confidence 0.75",
        "FIX_STRATEGY": [
            "Investigate test environment stability and resource availability during test runs",
            "Review and improve test suite timing and dependencies to reduce flakiness",
            "Implement retries with backoff as a temporary mitigation",
            "Enhance test logging to capture failure context for future analysis"
        ],
        "ROLLBACK_PLAN": "Not applicable as no deployment or configuration change caused the failure",
        "AUTO_FIX_FEASIBILITY": "Yes",
        "SEVERITY_LEVEL": "Low",
        "CONFIDENCE_SCORE": 0.75,
        "ERROR_COUNT": 1,
        "WARNING_COUNT": 1,
        "SUCCESS_INDICATORS": 1,
        "RESOLUTION_TIME": "~1 minute (time between failure and success on retry)",
        "BUSINESS_IMPACT": 0.1,
        "TECHNICAL_COMPLEXITY": "Low",
        "MONITORING_RECOMMENDATIONS": "Implement monitoring on test environment resource utilization and availability. Track frequency and patterns of test suite failures to identify trends. Alert on repeated test failures beyond retry threshold"
    },
    {
        "incident_id": 2,
        "EXECUTIVE_SUMMARY": "Deployment halted due to inability to connect to the production database server db-prod-01, causing a critical failure in the deployment process",
        "ROOT_CAUSE": "Database server db-prod-01 refused connection, likely due to network issues, server downtime, or configuration errors (confidence 0.85)",
        "FIX_STRATEGY": [
            "Verify database server status and network connectivity",
            "Check firewall and access control settings for db-prod-01",
            "Review recent changes to database or network configurations",
            "Restart database service if unresponsive",
            "Implement retry logic in deployment scripts for transient failures"
        ],
        "ROLLBACK_PLAN": "Revert deployment to last known good state before current deployment attempt. Confirm database connectivity before redeployment",
        "AUTO_FIX_FEASIBILITY": "No",
        "SEVERITY_LEVEL": "Critical",
        "CONFIDENCE_SCORE": 0.85,
        "ERROR_COUNT": 1,
        "WARNING_COUNT": 0,
        "SUCCESS_INDICATORS": 0,
        "RESOLUTION_TIME": "1-2 hours (dependent on database recovery)",
        "BUSINESS_IMPACT": 0.9,
        "TECHNICAL_COMPLEXITY": "Medium",
        "MONITORING_RECOMMENDATIONS": "Implement proactive database availability and connectivity monitoring with alerting. Monitor network latency and firewall rule changes affecting db-prod-01. Add deployment pre-checks for database connectivity before starting deployment"
    },
    {
        "incident_id": 3,
        "EXECUTIVE_SUMMARY": "Build process failed due to an invalid environment variable 'API_KEY', causing termination of the build",
        "ROOT_CAUSE": "Misconfigured or missing 'API_KEY' environment variable leading to build failure (confidence 0.95)",
        "FIX_STRATEGY": [
            "Validate and correct the 'API_KEY' environment variable value before build initiation",
            "Implement environment variable validation checks in CI pipeline"
        ],
        "ROLLBACK_PLAN": "Revert to last successful build configuration and environment variable settings to restore build functionality",
        "AUTO_FIX_FEASIBILITY": "Yes",
        "SEVERITY_LEVEL": "High",
        "CONFIDENCE_SCORE": 0.95,
        "ERROR_COUNT": 1,
        "WARNING_COUNT": 0,
        "SUCCESS_INDICATORS": 0,
        "RESOLUTION_TIME": "1-2 hours",
        "BUSINESS_IMPACT": 0.7,
        "TECHNICAL_COMPLEXITY": "Low",
        "MONITORING_RECOMMENDATIONS": "Implement environment variable validation monitoring in CI/CD pipeline. Alert on invalid or missing critical environment variables before build start"
    },
    {
        "incident_id": 4,
        "EXECUTIVE_SUMMARY": "A flaky test ('test_user_signup') caused intermittent failures requiring retries, but all tests ultimately passed after two retries",
        "ROOT_CAUSE": "Intermittent instability in the 'test_user_signup' test environment or test code causing flakiness (confidence 0.85)",
        "FIX_STRATEGY": [
            "Investigate and stabilize the 'test_user_signup' test to eliminate flakiness",
            "Review test dependencies and environment consistency",
            "Implement retries as a temporary mitigation",
            "Enhance test logging for better failure diagnostics"
        ],
        "ROLLBACK_PLAN": "Not applicable as no deployment or code change caused failure; issue is test instability",
        "AUTO_FIX_FEASIBILITY": "No",
        "SEVERITY_LEVEL": "Low",
        "CONFIDENCE_SCORE": 0.85,
        "ERROR_COUNT": 0,
        "WARNING_COUNT": 1,
        "SUCCESS_INDICATORS": 1,
        "RESOLUTION_TIME": "1-2 days (to identify and fix flaky test)",
        "BUSINESS_IMPACT": 0.1,
        "TECHNICAL_COMPLEXITY": "Medium",
        "MONITORING_RECOMMENDATIONS": "Implement test flakiness tracking dashboards. Alert on flaky test occurrences and retry counts. Monitor test environment stability metrics"
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())

prompt_scores = []
for prompt_idx, prompt_template in enumerate(prompts):
    all_scores = []
    for incident in GROUND_TRUTH:
        prompt = prompt_template + "\n\nINCIDENT_DATA:\n" + incident_to_flat_string(incident)
        # For ground truth comparison:
        response = llm.invoke(prompt)
        llm_output = normalize(response.content)
        gt_output = incident_to_flat_string(incident)
        score = semantic_similarity(gt_output, llm_output)
        all_scores.append(score)
    avg_prompt_score = np.mean(all_scores) if all_scores else 0
    prompt_scores.append((prompt_idx, avg_prompt_score))

print("\nPrompt Evaluation Results for CI/CD Analyzer:\n")
print("{:<10} {:<10}".format("Prompt ID", "Score"))
for idx, score in prompt_scores:
    print("{:<10} {:<10.4f}".format(idx, score))

best_prompt_idx, best_score = max(prompt_scores, key=lambda x: x[1])
print(f"\nBest Prompt: #{best_prompt_idx} (Score: {best_score:.4f})")

if best_score < 0.8:
    print("\n⚠️  Best score is below 0.8. Suggested Actionable Items to Improve Prompts:\n")
    for i, suggestion in enumerate([
        "Review clarity and specificity.",
        "Add or improve context.",
        "Refine structure.",
        "Encourage step-by-step reasoning.",
        "Balance brevity and detail.",
        "Include explicit evaluation criteria.",
        "Test with varied incidents.",
        "Iterate and compare."
    ], 1):
        print(f"{i}. {suggestion}")

print("\n--- Normalized Ground Truth String ---\n")
print(incident_to_flat_string(GROUND_TRUTH[0]))
