from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import google.generativeai as genai
import redis
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from crud_operations import CRUDOperations  # Import CRUD operations
import numpy as np  # For vector operations
from informational_agent import handle_informational_query  # Import informational agent
from action_agent import handle_action_command  # Import action agent

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Configure your Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Redis on port 9000
redis_client = redis.StrictRedis(host='localhost', port=9000, db=0, decode_responses=True)

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) 
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create a Pinecone index if it doesn't exist
if not pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")  # Adjust dimension based on your embedding model

# Connect to the Pinecone index
index = pinecone.Index(index_name)

# Initialize CRUD operations
crud = CRUDOperations()

@app.route('/')
def home():
    return render_template('index.html')

def classify_input(user_input):
    classification_prompt = ("Classify the user's input as either 'informational' or 'action_command.'"
    "User's Input: " + user_input + "\n\n"
    "Output format: 'informational' or 'action_command' ONLY."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(classification_prompt)
    return response.text

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')  # Get user ID from the request
    user_input = request.json.get('message')

    # Check for "exit" command
    if "exit" in user_input.lower():
        from ingest_data import ingest_data  # Import the ingest_data function
        ingest_data()  # Call the function to ingest data
        # Clear Redis database for the user
        crud.redis_delete(user_id)  # Clear user data from Redis
        return jsonify({'response': 'Goodbye!'})

    classification = classify_input(user_input)

    if "informational" in classification:
        response = handle_informational_query(user_input, user_id)
    elif "action_command" in classification:
        response = handle_action_command(user_input, user_id)
    else:
        response = "I don't understand. Please rephrase or ask something else."   
    
    # # Generate embeddings for the user input
    # embeddings = genai.embed_content(model="models/text-embedding-004", content=user_input)
    
    # # Retrieve conversation history from Redis
    # conversation_history = redis_client.get(user_id)
    # if conversation_history:
    #     conversation_history = json.loads(conversation_history)
    # else:
    #     conversation_history = []


    # # Prepare the context for the Gemini API
    # context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    # # Append the new user input to the conversation history
    # conversation_history.append({"role": "User", "content": user_input})

    # # Check for similar vectors in Pinecone
    # user_vector = embeddings['embedding']  # Assuming embeddings returns a dict with 'embedding'
    # similar_vectors = index.query(user_vector, top_k=3, include_metadata=True)  # Get top 5 similar vectors

    # # Prepare additional context from similar vectors
    # additional_context = []
    # for match in similar_vectors['matches']:
    #     additional_context.append(match['metadata']['text'])  # Assuming you stored content in metadata

    # # Query Supabase for facts
    # facts_response = crud.supabase_get_facts(user_id)
    # facts = facts_response.data  # Access the data attribute directly

    # # Generate vectors for each fact and calculate similarity
    # fact_vectors = []
    # for fact in facts:
    #     # Assuming fact_vector is stored as a string in the database, convert it to a NumPy array
    #     fact_vector = np.fromstring(fact['fact_vector'][1:-1], sep=',')  # Convert string to array
    #     fact_vectors.append((fact['fact'], fact_vector))

    # # Calculate similarities
    # similarities = []
    # for fact, fact_vector in fact_vectors:
    #     similarity = np.dot(user_vector, fact_vector)  # Calculate cosine similarity
    #     similarities.append((fact, similarity))

    # # Sort by similarity and get top 5 facts
    # top_facts = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # # Prepare the context for the Gemini API

    # # Combine context with additional context and top facts
    # full_context = (
    #     "System prompt: You are a helpful assistant. Answer the user's query based on the context provided."
    #     "THIS IS THE USER'S QUERY: " + user_input + "\n\n"
    #     "The following are additional contexts that might be relevant to the user's query."
    #     "If the any of the following data is not relevant to user's query, don't use it or bring it up."
    #     "Keep the response within 100 words.\n\n"
    #     "Current conversation till now: " + context + "\n"
    #     + "\nPotential Relevant facts: " + "\n".join([fact for fact, _ in top_facts])
    #     + "\nHistorical Relevant Context and Important Data: " + "\n".join(additional_context)
    # )

    # # Generate response from Gemini API
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # response = model.generate_content(full_context)
    
    # # Append the bot's response to the conversation history
    # conversation_history.append({"role": "Bot", "content": response.text})

    # # Store the updated conversation history back in Redis
    # redis_client.set(user_id, json.dumps(conversation_history))

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True) 