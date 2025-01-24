from crud_operations import CRUDOperations
import re
import google.generativeai as genai
import numpy as np

def handle_action_command(user_input, user_id):
    """Handle action commands from the user."""
    crud = CRUDOperations()

    # Use Gemini to extract intent and parameters
    extraction_prompt = (
        "Extract the action command and relevant parameters from the following user input:\n"
        f"User Input: {user_input}\n"
        "Output format: Action command, parameters as a dictionary."
        "Action command: forget, delete_fact, add_fact"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    extraction_response = model.generate_content(extraction_prompt)
    
    # Assuming the response is structured as "action_command: {command}, parameters: {params}"
    action_command, parameters = parse_extraction_response(extraction_response)

    if action_command == "forget":
        # Perform delete operations in all databases
        crud.redis_delete(user_id)  # Delete user data from Redis
        crud.pinecone_delete(user_id)  # Delete user vector from Pinecone
        # Delete all facts from Supabase for the user
        facts_response = crud.supabase_get_facts(user_id)
        for fact in facts_response.data:  # Access the data attribute directly
            crud.supabase_delete_fact(fact['id'])
        return {f'All data for user {user_id} has been forgotten.'}

    elif action_command == "delete_fact":
        fact_text = parameters.get("fact")
        if fact_text:
            # Generate a vector for the user's fact input
            fact_vector_response = genai.embed_content(model="models/text-embedding-004", content=fact_text)
            fact_vector = fact_vector_response['embedding']

            # Retrieve all facts from Supabase
            facts_response = crud.supabase_get_facts(user_id)
            facts = facts_response.data  # Access the data attribute directly

            # Check for similarity with each fact's vector
            for fact in facts:
                # Assuming fact_vector is stored as a string in the database, convert it to a NumPy array
                stored_vector = np.fromstring(fact['fact_vector'][1:-1], sep=',')  # Convert string to array
                similarity = np.dot(fact_vector, stored_vector)  # Calculate cosine similarity

                # Define a threshold for similarity (e.g., 0.9 for a strong match)
                if similarity > 0.8:  # Adjust threshold as needed
                    crud.supabase_delete_fact(fact['id'])
                    return {f'Fact "{fact_text}" has been deleted.'}

            return {f'No such fact exists.'}

    elif action_command == "add_fact":
        fact_text = parameters.get("fact")
        
        if fact_text:
            crud.supabase_insert_fact(user_id, fact_text)
            return {f'This fact has been added. Anything else?'}

    # Add more action commands as needed
    return {'Action command not recognized.'}

def parse_extraction_response(extraction_response):
    """Parse the response from Gemini to extract action command and parameters."""
    # This is a simple parsing logic; you may need to adjust it based on the actual response format
    response = extraction_response.text
    lines = response.splitlines()
    action_command = lines[0].split(":")[1].strip()  # Extract action command
    params = {}
    for line in lines[1:]:
        if "parameters" in line.lower():
            params_str = line.split(":", 1)[1].strip()  # Get the part after "Parameters:"
            params_str = params_str.strip("{} ")  # Remove curly braces and extra spaces
            for param in params_str.split(","):
                key, value = param.split(":")
                params[key.strip().strip("'")] = value.strip().strip("'")  # Extract parameters into a dictionary
    return action_command, params

def get_fact_id_by_text(user_id, fact_text):
    """Helper function to find the fact ID based on the fact text."""
    crud = CRUDOperations()
    facts_response = crud.supabase_get_facts(user_id)
    for fact in facts_response.data:
        if fact['fact'] == fact_text:
            return fact['id']
    return None 