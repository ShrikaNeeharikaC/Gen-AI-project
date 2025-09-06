import requests
import json

# List of questions to evaluate the RAG system
questions = [
    "Why did my Jenkins deploy fail?",
    "Why did my GitLab CI test fail intermittently?",
    # Add more questions as needed
]

eval_data = []

for q in questions:
    data = {"message": q, "session_id": "eval_session"}

    try:
        # Send POST request to the local chat API
        resp = requests.post("http://127.0.0.1:5004/api/chat", json=data)
        resp.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        result = resp.json()

        eval_data.append({
            "question": q,
            "contexts": result.get("contexts", []),
            "answer": result.get("response", ""),
            "ground_truth": ""  # Fill this manually with expected answers
        })

    except requests.exceptions.RequestException as e:
        print(f"Error for question: '{q}' → {e}")
        eval_data.append({
            "question": q,
            "contexts": [],
            "answer": "",
            "ground_truth": "",
            "error": str(e)
        })

# Save the evaluation data to a JSON file
try:
    with open("rag_eval_data.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    print("Evaluation data written to 'rag_eval_data.json'")
except IOError as io_err:
    print(f"Failed to write to file: {io_err}")
