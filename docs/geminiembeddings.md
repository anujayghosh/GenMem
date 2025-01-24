
Input token limit
2,048

Output dimension size
768

Sample code:
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

result = genai.embed_content(
        model="models/text-embedding-004",
        content="What is the meaning of life?")

print(str(result['embedding']))