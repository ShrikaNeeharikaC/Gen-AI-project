import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import AzureChatOpenAI
import os

# 1. Load ticket data
tickets = pd.read_excel('D:\\Innova Internship\\Book1.xlsx')

# 2. Define prompt templates
prompts = [
    f"""
You are a seasoned IT support analyst with comprehensive knowledge across all IT domains. 
Review the following support ticket:
Title: {{ticket.title}}
Description: {{ticket.description}}
Module: {{ticket.module}}

Provide the following:
- Summary (concise, 50–75 words)
- Priority (choose L1, L2, or L3 and indicate urgency: Critical, High, Medium, Low, or Planning)
- Category (select one: Core Services & Backend services, Product Development & UX, Platform & Infra, Data & System Management)
- Solution (one-sentence recommendation or fix)

Deliver precise, expert-level output. Avoid unnecessary commentary or formatting. Output only the requested content, exactly as specified.
""",

    f"""
You are an IT support expert. Based on the following ticket:
Title: {{ticket.title}}
Description: {{ticket.description}}
Module: {{ticket.module}}

Return only:
- Summary (50–75 words)
- Priority (L1, L2, or L3 + Critical/High/Medium/Low/Planning)
- Category (choose one: Core Services & Backend services, Product Development & UX, Platform & Infra, Data & System Management)
- Solution (one sentence)

Stick to facts. No extra content, formatting, or user interaction. Just the required fields.
""",

    f"""
As an expert in IT support, review this ticket carefully:
Title: {{ticket.title}}
Description: {{ticket.description}}
Module: {{ticket.module}}

Your output must include:
- Summary (about 50–75 words)
- Priority (L1, L2, or L3 and its severity: Critical, High, Medium, Low, or Planning)
- Category (choose one appropriate category: Core Services & Backend services, Product Development & UX, Platform & Infra, Data & System Management)
- Solution (single sentence fix or recommendation)

Be precise. No extra wording, explanations, or formatting. Respond with only what is asked above.
""",

    f"""
You are a domain expert in IT support. Analyze the ticket below:
Title: {{ticket.title}}
Description: {{ticket.description}}
Module: {{ticket.module}}

Provide the following items:
- Summary (approx. 50–75 words)
- Priority (L1, L2, or L3 and severity level: Critical, High, Medium, Low, or Planning)
- Category (select from: Core Services & Backend services, Product Development & UX, Platform & Infra, Data & System Management)
- Solution (one-sentence solution or recommendation)

Ensure your response is precise and free from extraneous content, formatting, or dialogue. Return only the items listed.
""",

    f"""
Act as an IT support professional. Ticket details:
Title: {{ticket.title}}
Description: {{ticket.description}}
Module: {{ticket.module}}

Respond with:
- Summary (50–75 words)
- Priority (L1, L2, or L3 + urgency: Critical, High, Medium, Low, or Planning)
- Category (pick one: Core Services & Backend services, Product Development & UX, Platform & Infra, Data & System Management)
- Solution (one sentence)

Output must be accurate and minimal—no extras, no formatting, no direct user communication.
"""
]

# 3. Define ground truth
ground_truth = {
    'DEF-0001': {
        'Summary': "Critical payroll calculation error in OrangeHRM system causing incorrect outputs and disrupting employee payment processing workflows, requiring immediate attention to prevent further impact on HR operations and employee satisfaction.",
        'Priority': "L1 - Critical",
        'Category': "Core Services & Backend services",
        'Solution': "Investigate and fix the calculation logic in the payroll processing engine, validate against test data, and implement automated regression testing."
    },
    'DEF-0002': {
        'Summary': "Leave management module displaying incorrect leave balance information to users, causing confusion in leave planning and potential approval workflow disruptions that affect employee scheduling and HR decision-making processes.",
        'Priority': "L1 - High",
        'Category': "Core Services & Backend services",
        'Solution': "Synchronize leave balance calculation logic with the database and implement real-time balance updates to ensure data consistency."
    },
    'DEF-0003': {
        'Summary': "Authentication system preventing users from accessing the OrangeHRM platform, blocking all HR operations and creating significant productivity loss until resolved, affecting all system users and business continuity.",
        'Priority': "L1 - Critical",
        'Category': "Platform & Infra",
        'Solution': "Debug authentication service, check database connectivity, and verify user credential validation logic to restore system access."
    },
    'DEF-0004': {
        'Summary': "Reporting module generating corrupted export files when users attempt to download data, compromising data integrity for analysis and external reporting requirements while limiting business intelligence capabilities.",
        'Priority': "L1 - Medium",
        'Category': "Data & System Management",
        'Solution': "Fix data serialization process in export functionality and implement data validation checks before file generation."
    },
    'DEF-0005': {
        'Summary': "Time tracking module preventing employees from submitting timesheets, blocking payroll processing workflows and creating delays in employee compensation and project time tracking for billing purposes.",
        'Priority': "L1 - High",
        'Category': "Core Services & Backend services",
        'Solution': "Resolve form validation issues and database transaction conflicts preventing timesheet data from being properly saved and processed."
    },
    'DEF-0006': {
        'Summary': "Notification system failing to send email alerts for important HR events, causing communication breakdowns and missed deadlines for approvals, reviews, and other time-sensitive HR processes.",
        'Priority': "L2 - Medium",
        'Category': "Platform & Infra",
        'Solution': "Check email service configuration, verify SMTP settings, and implement notification queue retry mechanisms for failed deliveries."
    },
    'DEF-0007': {
        'Summary': "User management system granting inappropriate access permissions to users, creating security risks and allowing unauthorized access to sensitive HR data and restricted system functionalities.",
        'Priority': "L2 - High",
        'Category': "Platform & Infra",
        'Solution': "Audit and fix role permission mappings, implement proper access control validation, and verify user role assignment logic."
    },
    'DEF-0008': {
        'Summary': "Performance management module failing to persist review scores and evaluation data, causing loss of important performance assessment information and disrupting employee development tracking and reporting.",
        'Priority': "L2 - Medium",
        'Category': "Core Services & Backend services",
        'Solution': "Fix database transaction handling in performance review forms and implement proper data persistence validation with error handling."
    },
    'DEF-0009': {
        'Summary': "Recruitment module causing complete system freeze when attempting to add new candidates, blocking hiring processes and preventing HR teams from managing job applications effectively.",
        'Priority': "L2 - High",
        'Category': "Product Development & UX",
        'Solution': "Identify and resolve memory leaks or infinite loops in candidate creation workflow and optimize form processing performance."
    },
    'DEF-0010': {
        'Summary': "Attendance tracking system not recording employee attendance logs properly, creating gaps in attendance data that affect payroll calculations and compliance reporting for workforce management.",
        'Priority': "L2 - High",
        'Category': "Data & System Management",
        'Solution': "Restore attendance logging functionality, implement data recovery procedures, and add monitoring to prevent future log loss."
    },
    'DEF-0011': {
        'Summary': "Dashboard interface displaying misaligned visual elements affecting user experience and readability, though not preventing core functionality, creating minor usability issues for daily system users.",
        'Priority': "L3 - Low",
        'Category': "Product Development & UX",
        'Solution': "Update CSS styling and layout configurations to fix alignment issues and ensure consistent responsive design across different screen sizes."
    },
    'DEF-0012': {
        'Summary': "Globalization module displaying incorrect date formats for different locales, causing confusion for international users and potentially affecting date-sensitive operations and data interpretation across regions.",
        'Priority': "L3 - Low",
        'Category': "Product Development & UX",
        'Solution': "Implement proper locale-based date formatting using internationalization libraries and validate format consistency across all modules."
    },
    'DEF-0013': {
        'Summary': "Directory search functionality returning inaccurate or irrelevant employee search results, reducing efficiency in finding staff information and impacting user productivity in employee lookup tasks.",
        'Priority': "L3 - Low",
        'Category': "Product Development & UX",
        'Solution': "Improve search algorithm relevance scoring and implement better indexing to ensure accurate employee directory search results."
    },
    'DEF-0014': {
        'Summary': "Help system containing non-functional links that prevent users from accessing support documentation, reducing self-service capabilities and potentially increasing support ticket volume for basic questions.",
        'Priority': "L3 - Medium",
        'Category': "Product Development & UX",
        'Solution': "Audit and update all help documentation links, implement link validation testing, and ensure proper help content accessibility."
    }
}

# 4. Azure OpenAI setup
os.environ["OPENAI_API_KEY"]="1eMZ4Wptb1r8d3JO66DFyEc1wxs9iZyDRXhA2y371SBZGX9PUNlyJQQJ99BFACYeBjFXJ3w3AAABACOGhuqi"
os.environ["AZURE_OPENAI_API_KEY"] = "1eMZ4Wptb1r8d3JO66DFyEc1wxs9iZyDRXhA2y371SBZGX9PUNlyJQQJ99BFACYeBjFXJ3w3AAABACOGhuqi"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://intern-openai-demo.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
# Optional: For embeddings if needed
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

# 5. Initialize AzureChatOpenAI client
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name="gpt-4.1-mini",  # Update to your deployment name if different
    temperature=0
)

# 6. Load embedding model
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
    # Implement a robust extraction logic based on your LLM output format
    # Example: split by lines and map to fields
    fields = {'Summary': '', 'Priority': '', 'Category': '', 'Solution': ''}
    lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
    current_field = None
    for line in lines:
        for field in fields:
            if line.lower().startswith(field.lower()):
                current_field = field
                fields[field] = line.split(':', 1)[-1].strip()
                break
        else:
            if current_field:
                fields[current_field] += ' ' + line
    return fields

# 7. Main loop: Evaluate each prompt version
prompt_scores = []
for prompt_idx, prompt_template in enumerate(prompts):
    all_scores = []
    for _, ticket in tickets.iterrows():
        ticket_id = ticket['ticket_id']
        # Prepare prompt
        prompt = prompt_template.format(ticket=ticket)
        # Get LLM response
        response = llm.invoke(prompt)
        llm_output = response.content
        # Extract fields
        llm_fields = extract_fields_from_llm_output(llm_output)
        gt_fields = ground_truth.get(ticket_id, {})
        # Evaluate similarity for each field
        field_scores = []
        for field in ['Summary', 'Priority', 'Category', 'Solution']:
            if field in gt_fields and gt_fields[field] and llm_fields.get(field):
                tfidf_score = tfidf_cosine(gt_fields[field], llm_fields[field])
                sem_score = semantic_similarity(gt_fields[field], llm_fields[field])
                avg_score = (tfidf_score + sem_score) / 2
                field_scores.append(avg_score)
        # Average score for this ticket
        if field_scores:
            all_scores.append(np.mean(field_scores))
    # Average score for this prompt version
    avg_prompt_score = np.mean(all_scores) if all_scores else 0
    prompt_scores.append((prompt_idx, avg_prompt_score))

# Print all prompt scores in a table format
print("\nPrompt Scores Table:\n")
print("{:<13} {:<13}".format("Prompt_Index", "Average_Score"))
for idx, score in prompt_scores:
    print("{:<13} {:<13.4f}".format(idx, score))

# Identify and print the best prompt
best_prompt_idx, best_score = max(prompt_scores, key=lambda x: x[1])
print(f"\nBest Prompt Version: {best_prompt_idx + 1} (Score: {best_score:.4f})\n")


# If best score is less than 0.8, generate a new prompt using ByteWave
if best_score < 0.8:
    try:
        from transformers import pipeline
        print("\nBest score is below 0.8. Generating a new prompt using ByteWave Prompt Generator...")
        generator = pipeline("text-generation", model="ByteWave/prompt-generator", 
        model_kwargs={"timeout": 60})
        generator = pipeline("text-generation", model="ByteWave/prompt-generator")
        act = """
        Action: IT Support Analyst
        Prompt: Review an IT support ticket and return summary (50–75 words), priority (L1/L2/L3 + urgency), category (choose from list), and solution (one sentence).
        """
        new_prompt_output = generator(act, do_sample=True, max_new_tokens=256)
        # The generated prompt is in new_prompt_output[0]['generated_text']
        generated_prompt = new_prompt_output[0]['generated_text']
        print("\nGenerated New Prompt:\n")
        print(generated_prompt)
        # Optionally, append to prompts list for next evaluation round
        prompts.append(generated_prompt)
        # Optionally, rerun evaluation loop here if you want to immediately test the new prompt
    except Exception as e:
        print("Prompt generation failed:", e)


# Save detailed results to a file
with open('prompt_scores_and_summary.txt', 'w', encoding='utf-8') as f:
    # Write scores table
    f.write("Prompt Evaluation Results\n")
    f.write("="*50 + "\n\n")
    f.write("Prompt Scores Table:\n")
    f.write("{:<13} {:<13}\n".format("Prompt_Index", "Average_Score"))
    for idx, score in prompt_scores:
        f.write("{:<13} {:<13.4f}\n".format(idx, score))
    
    # Write best prompt info
    f.write("\n" + "="*50 + "\n")
    f.write(f"Best Prompt Version: {best_prompt_idx + 1} (Score: {best_score:.4f})\n")
    f.write("="*50 + "\n\n")
    f.write("Best Prompt Text:\n")
    f.write("="*50 + "\n")
    f.write(prompts[best_prompt_idx])
    f.write("\n" + "="*50 + "\n\n")
    
    # Write action items
    f.write("Further Action Items:\n")
    f.write("- Review the best prompt and implement it in production workflows\n")
    f.write("- Analyze lower-scoring prompts for improvement opportunities\n")
    f.write("- Consider retesting with additional prompt variations\n")
    f.write("- Document findings in project documentation\n")
    
print("\nResults saved to 'best_prompt.txt' and 'prompt_scores_and_summary.txt'")

