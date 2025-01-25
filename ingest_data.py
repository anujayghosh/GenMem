import os
import redis
import json
from dotenv import load_dotenv
import pinecone
import google.generativeai as genai
from datetime import datetime
from supabase import create_client, Client  # Import Supabase client
from crud_operations import CRUDOperations

load_dotenv()

# Initialize Supabase
url = os.getenv("SUPABASE_URL")  # Your Supabase URL
key = os.getenv("SUPABASE_KEY")  # Your Supabase API key
supabase: Client = create_client(url, key)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
crud = CRUDOperations()

# Initialize Redis
redis_client = redis.StrictRedis(host='localhost', port=9000, db=0, decode_responses=True)

# Initialize Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) 
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create or connect to the Pinecone index
if not pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")  # Adjust dimension based on your embedding model

index = pinecone.Index(index_name)

# Function to summarize conversation and convert to vectors
def summarize_and_ingest(user_id):
    # Get conversation history
    conversation_history = redis_client.get(user_id)
    if conversation_history:
        conversation_history = json.loads(conversation_history)

        # Prepare the conversation for summarization
        full_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        model = genai.GenerativeModel("gemini-1.5-flash")
        # Summarize the conversation using Gemini
        summary_response = model.generate_content(f"Summarize the following conversation:\n{full_conversation}")
        summary = summary_response.text

        # Ask Gemini for learning insights based on the summary
        impdata_response = model.generate_content(f"Based on the following summary, find out all important data that can be used to answer the user's queries in the future. Keep the response within 100 words.\n{summary}")
        impdata_insights = impdata_response.text

        facts_response = model.generate_content(f"Based on the conversation, list out all facts given by the user that can be used to answer the user's queries in the future. Keep the response as a list of facts.\n{full_conversation}")
        facts_insights = facts_response.text  

        # Extract facts into a list
        facts_list = [fact.strip() for fact in facts_insights.split('\n') if fact.strip()]  # Split by new lines and clean up

        current_time = str(datetime.now())      
        # Combine summary and learning insights
        summarized_content = f"Timestamp: {current_time}, Summary: {summary}, Impdata: {impdata_insights}"

        # Convert the summarized content to a vector
        vector = genai.embed_content(model="models/text-embedding-004", content=summarized_content)

        # Upsert the vector with user ID as the ID
        crud.pinecone_upsert(user_id, vector['embedding'], {'text': summarized_content}) # Upsert the vector with user ID as the ID

        # Insert facts into Supabase
        for fact in facts_list:
            crud.supabase_insert_fact(user_id, fact)
            # factvector = genai.embed_content(model="models/text-embedding-004", content=fact)['embedding']
            # factv= "".join(factvector.numpy().tolist()[0])
            # supabase.table("semanticmem").insert({"user_id": user_id, "created_at": current_time, "fact": fact, "fact_vector": factvector}).execute()  # Insert each fact into the database

def ingest_data():
    # Retrieve all user IDs from Redis (or specify a user ID)
    user_ids = redis_client.keys()  # Get all keys (user IDs)

    for user_id in user_ids:
        summarize_and_ingest(user_id)

if __name__ == "__main__":
    ingest_data() 