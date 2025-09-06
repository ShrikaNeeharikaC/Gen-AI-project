import pandas as pd
import openai
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# 1. Load log data
logs = pd.read_excel('D:/Innova Internship/logs.xlsx')

# 2. Define prompt templates
prompts = [
    """
    Follow these steps:
    1. Identify the content type and main purpose.
    2. Extract key events, errors, warnings, and success indicators.
    3. Diagnose root cause(s) of any issues.
    4. Assess severity and business impact.
    5. Recommend specific fixes and preventive actions.

    Respond with a clear, structured analysis covering all steps. Be concise but thorough. Use bullet points or short paragraphs for each step.

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
]

# 3. Load ground truth
with open('D:/Innova Internship/log_ground_truth.json', 'r', encoding='utf-8') as f:
    ground_truth_list = json.load(f)
ground_truth = {entry['log_id']: entry['expected_analysis'] for entry in ground_truth_list}

# 4. Azure OpenAI setup
os.environ["AZURE_OPENAI_API_KEY"] = "1eMZ4Wptb1r8d3JO66DFyEc1wxs9iZyDRXhA2y371SBZGX9PUNlyJQQJ99BFACYeBjFXJ3w3AAABACOGhuqi"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://intern-openai-demo.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = "gpt-4-turbo"  # Your Azure deployment name

# 5. Load MiniVCC embedding model
minivcc_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your MiniVCC model if different

def tfidf_cosine(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf_matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def semantic_similarity(text1, text2):
    emb1 = minivcc_model.encode(text1, convert_to_tensor=True)
    emb2 = minivcc_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def extract_fields_from_llm_output(llm_output):
    fields = [
        'EXECUTIVE_SUMMARY',
        'ROOT_CAUSE',
        'FIX_STRATEGY',
        'ROLLBACK_PLAN',
        'AUTO_FIX_FEASIBILITY',
        'SEVERITY_LEVEL',
        'CONFIDENCE_SCORE',
        'ERROR_COUNT',
        'WARNING_COUNT',
        'SUCCESS_INDICATORS',
        'RESOLUTION_TIME',
        'BUSINESS_IMPACT',
        'TECHNICAL_COMPLEXITY',
        'MONITORING_RECOMMENDATIONS'
    ]
    result = {field: '' for field in fields}
    lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
    current_field = None
    for line in lines:
        for field in fields:
            if line.upper().startswith(field):
                current_field = field
                result[field] = line.split(':', 1)[-1].strip()
                break
        else:
            if current_field:
                result[current_field] += ' ' + line
    return result

# 6. Main evaluation loop
prompt_scores = []

for prompt_idx, prompt_template in enumerate(prompts):
    all_scores = []
    for _, log in logs.iterrows():
        log_id = log['log_id']
        prompt = prompt_template.format(log=log['content'], metadata=log.get('metadata', ''))
        
        # Get Azure OpenAI response
        try:
            response = openai.ChatCompletion.create(
                engine=azure_deployment,  # Azure uses 'engine' not 'model'
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            llm_output = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error processing log {log_id}: {str(e)}")
            continue
        
        llm_fields = extract_fields_from_llm_output(llm_output)
        gt_fields = ground_truth.get(log_id, {})
        
        field_scores = []
        for field in llm_fields.keys():
            if field in gt_fields and gt_fields[field] and llm_fields.get(field):
                # Numeric fields
                if field in ['ERROR_COUNT', 'WARNING_COUNT', 'SUCCESS_INDICATORS']:
                    try:
                        gt_val = int(gt_fields[field])
                        llm_val = int(llm_fields[field])
                        score = 1.0 if gt_val == llm_val else 0.0
                    except:
                        score = 0.0
                # Float fields
                elif field in ['CONFIDENCE_SCORE', 'BUSINESS_IMPACT']:
                    try:
                        gt_val = float(gt_fields[field])
                        llm_val = float(llm_fields[field])
                        score = 1.0 - min(abs(gt_val - llm_val), 1.0)
                    except:
                        score = 0.0
                # Categorical fields
                elif field in ['SEVERITY_LEVEL', 'TECHNICAL_COMPLEXITY', 'AUTO_FIX_FEASIBILITY']:
                    score = 1.0 if gt_fields[field].strip().lower() == llm_fields[field].strip().lower() else 0.0
                # Text fields (using MiniVCC)
                else:
                    try:
                        tfidf_score = tfidf_cosine(str(gt_fields[field]), str(llm_fields[field]))
                        sem_score = semantic_similarity(str(gt_fields[field]), str(llm_fields[field]))
                        score = (tfidf_score + sem_score) / 2
                    except:
                        score = 0.0
                field_scores.append(score)
        
        if field_scores:
            all_scores.append(np.mean(field_scores))
    
    avg_prompt_score = np.mean(all_scores) if all_scores else 0
    prompt_scores.append((prompt_idx, avg_prompt_score))

# 7. Identify best prompt
best_prompt_idx, best_score = max(prompt_scores, key=lambda x: x[1])
print(f"\nBest Prompt Version: {best_prompt_idx + 1} (Score: {best_score:.4f})")
print("\nPrompt Text:")
print(prompts[best_prompt_idx])
