# rag_evaluation.py - Updated with Ragas integration
import json
import time
import base64
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_community.llms import Ollama
from elasticsearch import Elasticsearch
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

@dataclass
class RagasEvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_recall_at_k: float  # Added for context recall @k

class CICDRagEvaluator:
    def __init__(self, chatbot=None):
        """Initialize evaluator with your existing chatbot instance or create new ES connection"""
        if chatbot:
            self.chatbot = chatbot
            self.es = chatbot.es
            self.knowledge_index = chatbot.knowledge_index
        else:
            # Use your exact Elasticsearch Cloud configuration
            api_key_decoded = base64.b64decode("SEtWQlVKY0JRLUE2QldTNnB3c0U6TXJ6a1dKZ0xIQ01fTndYNWtLRVhhdw==").decode('utf-8')
            key_parts = api_key_decoded.split(':')
            self.api_key_id = key_parts[0]
            self.api_key_secret = key_parts[1]
            
            # Initialize Elasticsearch with your exact configuration
            self.es = Elasticsearch(
                ["https://a705a31d6c434d5d9b8801b99d0ef7f7.us-central1.gcp.cloud.es.io"],
                api_key=(self.api_key_id, self.api_key_secret),
                verify_certs=True,
                request_timeout=60
            )
            self.knowledge_index = "cicd_knowledge_base"
            
        self.llm = Ollama(
            model="llama3.1:8b",
            temperature=0.1,
            num_ctx=4096
        )
        
    def get_knowledge_base_from_elasticsearch(self) -> List[Dict]:
        """Retrieve all knowledge base entries from your Elasticsearch Cloud"""
        try:
            print("🔍 Retrieving knowledge base from Elasticsearch Cloud...")
            
            # Query to get all knowledge base entries
            query = {
                "size": 100,
                "query": {"match_all": {}},
                "_source": [
                    "title", "content", "error_type", "technology", 
                    "stage", "severity", "solution", "commands", "tags"
                ]
            }
            
            response = self.es.search(index=self.knowledge_index, body=query)
            
            knowledge_entries = []
            for hit in response['hits']['hits']:
                knowledge_entry = hit['_source']
                knowledge_entry['_id'] = hit['_id']  # Add document ID
                knowledge_entries.append(knowledge_entry)
            
            print(f"✅ Retrieved {len(knowledge_entries)} knowledge base entries")
            return knowledge_entries
            
        except Exception as e:
            print(f"❌ Error retrieving knowledge base: {e}")
            return []
    
    def generate_qa_pairs_from_knowledge(self) -> List[Dict]:
        """Generate QA pairs from your existing knowledge base"""
        qa_pairs = []
        knowledge_entries = self.get_knowledge_base_from_elasticsearch()
        
        if not knowledge_entries:
            print("❌ No knowledge base entries found!")
            return []
        
        print(f"📝 Generating QA pairs from {len(knowledge_entries)} entries...")
        
        for i, knowledge in enumerate(knowledge_entries):
            print(f"  Processing entry {i+1}/{len(knowledge_entries)}: {knowledge['title'][:50]}...")
            qa_pairs.extend(self._generate_questions_for_knowledge(knowledge))
        
        print(f"✅ Generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def _generate_questions_for_knowledge(self, knowledge: Dict) -> List[Dict]:
        """Generate different types of questions for each knowledge entry"""
        qa_generation_prompt = f"""Based on this CI/CD knowledge, generate 3 question types:
KNOWLEDGE ENTRY:
Title: {knowledge.get('title', 'N/A')}
Content: {knowledge.get('content', 'N/A')}
Technology: {knowledge.get('technology', 'N/A')}
Error Type: {knowledge.get('error_type', 'N/A')}
Stage: {knowledge.get('stage', 'N/A')}
Severity: {knowledge.get('severity', 'N/A')}
Solution: {knowledge.get('solution', 'N/A')}
Commands: {knowledge.get('commands', 'N/A')}

Generate exactly 3 questions in this format:
QUESTION 1 (Diagnostic): [Diagnostic question]
EXPECTED_ANSWER 1: [Expected answer]
QUESTION 2 (Solution): [Solution question]
EXPECTED_ANSWER 2: [Expected answer]
QUESTION 3 (Prevention): [Prevention question]
EXPECTED_ANSWER 3: [Expected answer]"""

        try:
            response = self.llm.invoke(qa_generation_prompt)
            return self._parse_generated_qa(response, knowledge)
        except Exception as e:
            print(f"Error generating QA: {e}")
            return []
    
    def _parse_generated_qa(self, response: str, knowledge: Dict) -> List[Dict]:
        """Parse the generated QA pairs"""
        qa_pairs = []
        sections = response.split("QUESTION")
        
        for i, section in enumerate(sections[1:4]):  # Process exactly 3 questions
            if "EXPECTED_ANSWER" in section:
                parts = section.split("EXPECTED_ANSWER")
                if len(parts) >= 2:
                    question = parts[0].split(":")[1].strip() if ":" in parts[0] else parts[0].strip()
                    expected_answer = parts[1].split(":")[1].strip() if ":" in parts[1] else parts[1].strip()
                    
                    if question and expected_answer:
                        qa_pairs.append({
                            "question": question,
                            "expected_answer": expected_answer,
                            "knowledge_source": knowledge.get('title', 'Unknown'),
                            "technology": knowledge.get('technology', 'Unknown'),
                            "error_type": knowledge.get('error_type', 'Unknown'),
                            "stage": knowledge.get('stage', 'Unknown'),
                            "severity": knowledge.get('severity', 'Unknown'),
                            "question_type": f"type_{i+1}",
                            "knowledge_id": knowledge.get('_id', 'unknown')
                        })
        return qa_pairs
    
    def test_chatbot_with_question(self, question: str, session_id: str) -> Dict:
        """Test chatbot and return response with contexts"""
        if self.chatbot:
            result = self.chatbot.generate_response(question, session_id, "evaluator")
            # Assume chatbot returns contexts in response
            return result
        else:
            # Simulate response with contexts
            return {
                "response": f"Test response for: {question}",
                "contexts": [f"Context for {question}"],  # Simulated contexts
                "session_id": session_id
            }
    
    def run_ragas_evaluation(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """Evaluate using Ragas metrics"""
        print("🔬 Starting Ragas Evaluation")
        
        # Collect data for Ragas
        questions = []
        ground_truths = []
        answers = []
        contexts_list = []
        
        for i, qa_pair in enumerate(qa_pairs):
            print(f"  Testing {i+1}/{len(qa_pairs)}: {qa_pair['question'][:60]}...")
            session_id = f"eval_session_{i}"
            chatbot_result = self.test_chatbot_with_question(qa_pair['question'], session_id)
            
            questions.append(qa_pair['question'])
            ground_truths.append(qa_pair['expected_answer'])
            answers.append(chatbot_result['response'])
            contexts_list.append(chatbot_result.get('contexts', []))
        
        # Create Ragas dataset
        rag_dataset = Dataset.from_dict({
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": contexts_list
        })
        
        # Define Ragas metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        # Run evaluation
        result = evaluate(rag_dataset, metrics=metrics)
        
        return {
            "dataset": rag_dataset,
            "result": result,
            "scores": result.to_pandas().mean().to_dict()
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation of the RAG chatbot"""
        print("🔬 Starting Comprehensive RAG Evaluation")
        
        # Test Elasticsearch connection
        try:
            health = self.es.cluster.health()
            print(f"✅ Elasticsearch connection: {health['status']}")
        except Exception as e:
            print(f"❌ Elasticsearch connection failed: {e}")
            return {}
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs_from_knowledge()
        if not qa_pairs:
            print("❌ No QA pairs generated")
            return {}
        
        # Run Ragas evaluation
        ragas_results = self.run_ragas_evaluation(qa_pairs)
        
        # Generate report
        report = self._generate_evaluation_report(ragas_results)
        
        return {
            "ragas_results": ragas_results,
            "report": report,
            "qa_pairs": qa_pairs
        }
    
    def _generate_evaluation_report(self, results: Dict) -> str:
        """Generate evaluation report based on Ragas results"""
        scores = results['scores']
        
        report = "\n" + "="*80 + "\n"
        report += "🔬 RAGAS EVALUATION REPORT\n"
        report += "="*80 + "\n"
        
        # Metric Scores
        report += f"\n📊 RAGAS METRIC SCORES\n" + "-"*50 + "\n"
        report += f"Faithfulness: {scores['faithfulness']:.4f}\n"
        report += f"Answer Relevancy: {scores['answer_relevancy']:.4f}\n"
        report += f"Context Precision: {scores['context_precision']:.4f}\n"
        report += f"Context Recall: {scores['context_recall']:.4f}\n"
        
        # Interpretation
        report += f"\n💡 INTERPRETATION\n" + "-"*50 + "\n"
        report += "• Faithfulness: Measures factual consistency between answer and context\n"
        report += "• Answer Relevancy: Assesses how well the answer addresses the question\n"
        report += "• Context Precision: Evaluates ranking of relevant context items\n"
        report += "• Context Recall: Measures alignment between context and ground truth\n"
        
        # Recommendations
        report += f"\n🚀 RECOMMENDATIONS\n" + "-"*50 + "\n"
        if scores['faithfulness'] < 0.8:
            report += "• Improve answer grounding in context (reduce hallucinations)\n"
        if scores['answer_relevancy'] < 0.8:
            report += "• Enhance answer focus on question intent\n"
        if scores['context_precision'] < 0.8:
            report += "• Optimize retrieval to rank relevant context higher\n"
        if scores['context_recall'] < 0.8:
            report += "• Improve retrieval of all relevant context elements\n"
        
        report += "\n" + "="*80 + "\n"
        return report

def run_rag_evaluation():
    """Run the RAG evaluation"""
    try:
        from app import chatbot
        print("✅ Using chatbot from app.py")
        evaluator = CICDRagEvaluator(chatbot)
    except ImportError:
        print("⚠️ Using direct Elasticsearch connection")
        evaluator = CICDRagEvaluator()
    
    evaluation_results = evaluator.run_comprehensive_evaluation()
    
    if not evaluation_results:
        print("❌ Evaluation failed")
        return
    
    # Print and save results
    print(evaluation_results['report'])
    with open('ragas_evaluation_results.json', 'w') as f:
        json.dump({
            "scores": evaluation_results['ragas_results']['scores'],
            "qa_pairs": evaluation_results.get('qa_pairs', [])
        }, f, indent=2)
    
    print("📁 Results saved to ragas_evaluation_results.json")

if __name__ == "__main__":
    run_rag_evaluation()
