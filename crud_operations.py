import os
import redis
import pinecone
from supabase import create_client, Client
import google.generativeai as genai
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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

# Initialize Supabase
url = os.getenv("SUPABASE_URL")  # Your Supabase URL
key = os.getenv("SUPABASE_KEY")  # Your Supabase API key
supabase: Client = create_client(url, key)

class CRUDOperations:
    def __init__(self):
        self.redis_client = redis_client
        self.pinecone_index = index
        self.supabase_client = supabase

    # Redis Operations
    def redis_set(self, user_id, data):
        """Set data in Redis for a specific user ID."""
        self.redis_client.set(user_id, data)

    def redis_get(self, user_id):
        """Get data from Redis for a specific user ID."""
        return self.redis_client.get(user_id)

    def redis_delete(self, user_id):
        """Delete data from Redis for a specific user ID."""
        self.redis_client.delete(user_id)

    # Pinecone Operations
    def pinecone_upsert(self, user_id, vector, metadata):
        """Upsert a vector into Pinecone."""
        self.pinecone_index.upsert([(user_id, vector, metadata)])

    def pinecone_query(self, vector, top_k=3):
        """Query Pinecone for similar vectors."""
        return self.pinecone_index.query(vector, top_k=top_k, include_metadata=True)

    def pinecone_delete(self, user_id):
        """Delete a vector from Pinecone by user ID."""
        self.pinecone_index.delete(ids=[user_id])

    # Supabase Operations
    def supabase_insert_fact(self, user_id, fact):
        """Insert a fact into the Supabase database."""
        current_time = str(datetime.now())
        factvector = genai.embed_content(model="models/text-embedding-004", content=fact)['embedding']
        self.supabase_client.table("semanticmem").insert({
            "user_id": user_id,
            "created_at": current_time,
            "fact": fact,
            "fact_vector": factvector
        }).execute()

    def supabase_get_facts(self, user_id):
        """Get all facts for a specific user ID from Supabase."""
        return self.supabase_client.table("semanticmem").select("*").eq("user_id", user_id).execute()

    def supabase_delete_fact(self, fact_id):
        """Delete a fact from Supabase by fact ID."""
        self.supabase_client.table("semanticmem").delete().eq("id", fact_id).execute()

# Example usage
if __name__ == "__main__":
    crud = CRUDOperations()
    # Example operations
    user_id = "user123"
    data = {"role": "User", "content": "Hello, how are you?"}
    
    # Redis operations
    crud.redis_set(user_id, data)
    print(crud.redis_get(user_id))
    crud.redis_delete(user_id)

    # Pinecone operations
    vector = [0.1, 0.2, 0.3]  # Example vector
    metadata = {"summary": "Example summary"}
    crud.pinecone_upsert(user_id, vector, metadata)
    print(crud.pinecone_query(vector))

    # Supabase operations
    crud.supabase_insert_fact(user_id, "This is a fact.")
    print(crud.supabase_get_facts(user_id)) 