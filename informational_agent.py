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

def handle_informational_query(user_input, user_id):
    """Handle informational queries from the user."""
    crud = CRUDOperations()

    # Generate embeddings for the user input
    embeddings = genai.embed_content(model="models/text-embedding-004", content=user_input)
    user_vector = embeddings['embedding']  # Assuming embeddings returns a dict with 'embedding'

    # Retrieve conversation history from Redis
    conversation_history = redis_client.get(user_id)
    if conversation_history:
        conversation_history = json.loads(conversation_history)
    else:
        conversation_history = []

    # Append the new user input to the conversation history
    conversation_history.append({"role": "User", "content": user_input})

    # Check for similar vectors in Pinecone
    similar_vectors = index.query(user_vector, top_k=3, include_metadata=True)  # Get top 3 similar vectors

    # Prepare additional context from similar vectors
    additional_context = []
    for match in similar_vectors['matches']:
        additional_context.append(match['metadata']['text'])  # Assuming you stored content in metadata

    # Query Supabase for facts
    facts_response = crud.supabase_get_facts(user_id)
    facts = facts_response.data  # Access the data attribute directly

    # Generate vectors for each fact and calculate similarity
    fact_vectors = []
    for fact in facts:
        fact_vector = np.fromstring(fact['fact_vector'][1:-1], sep=',')  # Convert string to array
        fact_vectors.append((fact['fact'], fact_vector))

    # Calculate similarities
    similarities = []
    for fact, fact_vector in fact_vectors:
        similarity = np.dot(user_vector, fact_vector)  # Calculate cosine similarity
        similarities.append((fact, similarity))

    # Sort by similarity and get top 5 facts
    top_facts = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # Prepare the context for the Gemini API
    full_context = (
        "System prompt: You are a helpful assistant. Answer the user's query based on the context provided."
        "THIS IS THE USER'S QUERY: " + user_input + "\n\n"
        "The following are additional contexts that might be relevant to the user's query."
        "If any of the following data is not relevant to the user's query, don't use it or bring it up."
        "Keep the response within 100 words.\n\n"
        "Current conversation till now: " + "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history]) + "\n"
        + "\nPotential Relevant facts: " + "\n".join([fact for fact, _ in top_facts])
    )

    # Generate response from Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(full_context)

    # Append the bot's response to the conversation history
    conversation_history.append({"role": "Bot", "content": response.text})

    # Store the updated conversation history back in Redis
    redis_client.set(user_id, json.dumps(conversation_history))

    return response.text