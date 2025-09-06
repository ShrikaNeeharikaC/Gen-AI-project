from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG Chatbot service URL (assuming it's running on port 5004)
CHATBOT_SERVICE_URL = "http://localhost:5004"

@app.route('/chat', methods=['POST'])
def chat():
    """Proxy endpoint for chatbot communication"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Forward request to RAG chatbot service
        chatbot_response = requests.post(
            f"{CHATBOT_SERVICE_URL}/chat",
            json={
                "message": message,
                "session_id": session_id
            },
            timeout=30
        )
        
        if chatbot_response.status_code == 200:
            response_data = chatbot_response.json()
            
            # Format response for UI
            formatted_response = {
                "message": response_data.get('response', 'No response'),
                "intent": response_data.get('intent', 'unknown'),
                "confidence": response_data.get('confidence', 0.8),
                "session_id": response_data.get('session_id', session_id),
                "timestamp": response_data.get('timestamp'),
                "model": response_data.get('model', 'llama3.1:8b'),
                "knowledge_sources": response_data.get('knowledge_sources', []),
                "relevant_knowledge": response_data.get('relevant_knowledge', 0)
            }
            
            return jsonify(formatted_response)
        else:
            return jsonify({"error": "Chatbot service unavailable"}), 503
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Chatbot service timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to chatbot service"}), 503
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Check if chatbot service is available
        health_response = requests.get(f"{CHATBOT_SERVICE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            return jsonify({"status": "healthy", "chatbot_service": "connected"})
        else:
            return jsonify({"status": "degraded", "chatbot_service": "disconnected"}), 503
    except:
        return jsonify({"status": "degraded", "chatbot_service": "disconnected"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
