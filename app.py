from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Configure your Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(user_input)
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True) 