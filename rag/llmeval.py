import json
import requests
import openai
import os
from sentence_transformers import SentenceTransformer, util

# === CONFIGURATION ===
GROUND_TRUTH_FILE = "../llm_ground_truth.json"
OPENAI_MODEL = "gpt-4-turbo"  # You can change to any OpenAI chat model that supports JSON output
GEMINI_ENDPOINT = "http://localhost:5004/analyze"

# Set your OpenAI API key as an environment variable before running this script
openai.api_key = os.environ["OPENAI_API_KEY"]

# === LOAD SENTENCE TRANSFORMER MODEL ===
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(a, b):
    """Compute semantic similarity between two strings using sentence-transformers."""
    a_str = str(a) if not isinstance(a, str) else a
    b_str = str(b) if not isinstance(b, str) else b
    emb_a = model.encode(a_str, convert_to_tensor=True)
    emb_b = model.encode(b_str, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb_a, emb_b).item()

def analyze_with_openai(content, metadata):
    """Get analysis from OpenAI API, expecting JSON output."""
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in log analysis and failure diagnosis."},
                {"role": "user", "content": f"""Analyze the following log and metadata. 
Return a JSON object with keys: failure_summary, root_cause, fix_suggestion.

Log content:
{content}

Metadata:
{metadata}
"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message['content'])
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {
            "failure_summary": "Analysis failed",
            "root_cause": "API error",
            "fix_suggestion": "Check API configuration"
        }

def main():
    # Load test cases
    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)

    openai_scores, gemini_scores = [], []

    for i, case in enumerate(test_cases):
        payload = {
            "content": case["content"],
            "metadata": case["metadata"]
        }

        # --- OpenAI Analysis ---
        openai_out = analyze_with_openai(case["content"], case["metadata"])

        # --- Gemini Analysis ---
        gemini_resp = requests.post(GEMINI_ENDPOINT, json=payload)
        gemini_out = gemini_resp.json()["results"]["final_analysis"]

        gt = case["expected_analysis"]

        # --- Semantic Similarity ---
        openai_summary_sim = semantic_similarity(openai_out.get("failure_summary", ""), gt.get("failure_summary", ""))
        openai_root_sim = semantic_similarity(openai_out.get("root_cause", ""), gt.get("root_cause", ""))
        openai_fix_sim = semantic_similarity(openai_out.get("fix_suggestion", ""), gt.get("fix_suggestion", ""))

        gemini_summary_sim = semantic_similarity(gemini_out.get("failure_summary", ""), gt.get("failure_summary", ""))
        gemini_root_sim = semantic_similarity(gemini_out.get("root_cause", ""), gt.get("root_cause", ""))
        gemini_fix_sim = semantic_similarity(gemini_out.get("fix_suggestion", ""), gt.get("fix_suggestion", ""))

        openai_scores.append((openai_summary_sim, openai_root_sim, openai_fix_sim))
        gemini_scores.append((gemini_summary_sim, gemini_root_sim, gemini_fix_sim))

        # --- Print Results ---
        print(f"\nTest {i+1}:")
        print(f"Question (Log):\n{case['content']}")
        print(f"OpenAI Answer:\n{openai_out}")
        print(f"Gemini Answer:\n{gemini_out}")
        print(f"Ground Truth:\n{gt}")

        print(f"Scores:")
        print(f"  OpenAI - Summary={openai_summary_sim:.2f}, Root={openai_root_sim:.2f}, Fix={openai_fix_sim:.2f}")
        print(f"  Gemini - Summary={gemini_summary_sim:.2f}, Root={gemini_root_sim:.2f}, Fix={gemini_fix_sim:.2f}")

    # --- Aggregate/Average Scores ---
    openai_avg = [sum(x)/len(x) for x in zip(*openai_scores)]
    gemini_avg = [sum(x)/len(x) for x in zip(*gemini_scores)]

    print("\n=== Average Semantic Similarity Scores ===")
    print(f"OpenAI - Summary={openai_avg[0]:.2f}, Root={openai_avg[1]:.2f}, Fix={openai_avg[2]:.2f}")
    print(f"Gemini - Summary={gemini_avg[0]:.2f}, Root={gemini_avg[1]:.2f}, Fix={gemini_avg[2]:.2f}")

    # --- Best Model Per Field ---
    for i, field in enumerate(["Summary", "Root Cause", "Fix Suggestion"]):
        if openai_avg[i] > gemini_avg[i]:
            winner = "OpenAI"
        elif gemini_avg[i] > openai_avg[i]:
            winner = "Gemini"
        else:
            winner = "Tie"
        print(f"{field}: Best = {winner}")

if __name__ == "__main__":
    main()
