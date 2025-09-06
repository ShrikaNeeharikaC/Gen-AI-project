import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pd.set_option('display.max_colwidth', None)  # Show full text in columns
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.width', 1000) 

# ---------- Configure Azure OpenAI for GPT-4.1 mini ----------
os.environ["AZURE_OPENAI_API_KEY"] = "1eMZ4Wptb1r8d3JO66DFyEc1wxs9iZyDRXhA2y371SBZGX9PUNlyJQQJ99BFACYeBjFXJ3w3AAABACOGhuqi"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://intern-openai-demo.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
# Fallback key for embeddings if needed
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

# ---------- Load Azure LLM ----------
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name="gpt-4.1-mini",
    temperature=0
)

# ---------- Use SentenceTransformer Embeddings directly ----------


# ---------- Sample RAG evaluation data ----------
samples = [
        {
            "question": "How do I fix Docker build failed with exit code 125?",
            "answer": "Docker exit code 125 typically indicates a container runtime error. Check your Dockerfile syntax, ensure base images are available, verify port bindings aren't conflicting, and check for permission issues.",
            "contexts": [
                "Docker exit code 125 means the docker run command itself failed. This is usually due to invalid arguments or configuration issues.",
                "Common causes include: invalid Dockerfile syntax, missing base images, port conflicts, or permission denied errors.",
                "To troubleshoot: check docker logs, verify Dockerfile syntax, ensure images exist, and check port availability."
            ],
            "ground_truth": "Docker exit code 125 indicates a container runtime failure. Check Dockerfile syntax, verify base images, resolve port conflicts, and fix permission issues."
        },
        {
            "question": "What causes Kubernetes ImagePullBackOff error?",
            "answer": "ImagePullBackOff occurs when Kubernetes cannot pull a container image. Common causes include: incorrect image name/tag, private registry authentication issues, network connectivity problems, or image doesn't exist.",
            "contexts": [
                "ImagePullBackOff is a Kubernetes error state when the kubelet cannot pull the specified container image.",
                "Common causes: wrong image name, missing authentication for private registries, network issues, or non-existent images.",
                "Solutions: verify image name and tag, check registry credentials, ensure network connectivity, and validate image exists."
            ],
            "ground_truth": "ImagePullBackOff happens when Kubernetes can't pull container images due to wrong names, authentication issues, network problems, or missing images."
        },
        {
            "question": "How to troubleshoot Jenkins build failures?",
            "answer": "To troubleshoot Jenkins build failures: check console output for errors, verify workspace permissions, ensure required plugins are installed, check SCM connectivity, and validate build scripts syntax.",
            "contexts": [
                "Jenkins build failures can be diagnosed through console output, which shows detailed error messages and stack traces.",
                "Common issues include: workspace permission problems, missing plugins, SCM connection failures, and script syntax errors.",
                "Best practices: review console logs, check plugin compatibility, verify credentials, and test scripts locally."
            ],
            "ground_truth": "Troubleshoot Jenkins failures by checking console output, verifying permissions, ensuring plugins are installed, and validating SCM connectivity."
        },
        {
            "question": "Why is my CI/CD pipeline failing intermittently?",
            "answer": "Intermittent CI/CD failures often result from: flaky tests, network timeouts, resource constraints, race conditions, external service dependencies, or environment inconsistencies.",
            "contexts": [
                "Intermittent failures are challenging to debug as they don't occur consistently and may be environment-dependent.",
                "Common causes: unstable tests, network issues, insufficient resources, timing problems, and external dependencies.",
                "Solutions: identify flaky tests, implement retries, monitor resources, stabilize environments, and mock external services."
            ],
            "ground_truth": "Intermittent pipeline failures are caused by flaky tests, network issues, resource constraints, race conditions, and external dependencies."
        },
        {
            "question": "How to resolve npm build errors in GitHub Actions?",
            "answer": "To fix npm build errors in GitHub Actions: ensure correct Node.js version, clear npm cache, check package.json dependencies, verify environment variables, and use npm ci instead of npm install.",
            "contexts": [
                "GitHub Actions npm build errors often relate to Node.js version mismatches or dependency resolution issues.",
                "Common solutions: specify Node.js version, use npm ci for consistent installs, clear cache, and check for missing dependencies.",
                "Best practices: pin Node.js versions, use package-lock.json, set proper environment variables, and handle private packages."
            ],
            "ground_truth": "Fix npm build errors by using correct Node.js version, clearing cache, using npm ci, and verifying dependencies."
        }
    ]
    


# ---------- Prepare dataset ----------
ragas_dict = {
    "question": [],
    "contexts": [],
    "answer": [],
    "ground_truth": []
}

for sample in samples:
    if sample["contexts"] and sample["answer"].strip() and sample["ground_truth"].strip():
        ragas_dict["question"].append(sample["question"])
        ragas_dict["contexts"].append(sample["contexts"])
        ragas_dict["answer"].append(sample["answer"])
        ragas_dict["ground_truth"].append(sample["ground_truth"])

ragas_dataset = Dataset.from_dict(ragas_dict)

# ---------- Evaluate using Ragas ----------
ragas_results = evaluate(
    ragas_dataset,
    metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    llm=llm,
    embeddings=embedding  # ✅ Custom embedding
)
ragas_results=ragas_results.to_pandas()

# ---------- Show results ----------
#print(ragas_results.to_pandas())import pandas as pd

# Assuming result_ragas is your DataFrame
for idx, row in ragas_results.iterrows():
    print(f"\n{'='*80}")
    print(f"🧪 Sample {idx + 1}")
    print(f"{'='*80}")
    
    print(f"📝 User Input:\n{row['user_input']}\n")
    
    print(f"📚 Retrieved Contexts:")
    for i, ctx in enumerate(row['retrieved_contexts']):
        print(f"  {i+1}. {ctx[:200]}{'...' if len(ctx) > 200 else ''}")  # Shorten for readability
    print()
    
    print(f"💬 Response (truncated):\n{row['response'][:700]}{'...' if len(row['response']) > 700 else ''}\n")
    
    print(f"📘 Reference:\n{row['reference'][:500]}{'...' if len(row['reference']) > 500 else ''}\n")
    
    print(f"📊 RAGAS Scores:")
    print(f"  - Answer Relevancy : {row['answer_relevancy']:.3f}")
    print(f"  - Faithfulness     : {row['faithfulness']:.3f}")
    print(f"  - Context Precision: {row['context_precision']:.3f}")
    print(f"  - Context Recall    : {row['context_recall']:.3f}")
    print()