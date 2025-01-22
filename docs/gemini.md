Install the Gemini API library
Using Python 3.9+, install the google-generativeai package using the following pip command:


pip install -q -U google-generativeai
Make your first request
Get a Gemini API key in Google AI Studio

Use the generateContent method to send a request to the Gemini API.


import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "Explain how AI works"}]
    }]
   }'