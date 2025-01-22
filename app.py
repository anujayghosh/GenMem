from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import google.generativeai as genai
import redis
import json

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Configure your Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Redis on port 9000
redis_client = redis.StrictRedis(host='localhost', port=9000, db=0, decode_responses=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')  # Get user ID from the request
    user_input = request.json.get('message')

    # Retrieve conversation history from Redis
    conversation_history = redis_client.get(user_id)
    if conversation_history:
        conversation_history = json.loads(conversation_history)
    else:
        conversation_history = []

    # Append the new user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Prepare the context for the Gemini API
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    # Generate response from Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(context)
    
    # Append the bot's response to the conversation history
    conversation_history.append({"role": "bot", "content": response.text})

    # Store the updated conversation history back in Redis
    redis_client.set(user_id, json.dumps(conversation_history))

    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True) 