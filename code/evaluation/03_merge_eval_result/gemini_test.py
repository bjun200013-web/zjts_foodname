import os
from google import genai



os.environ["GEMINI_API_KEY"] = 'AIzaSyBureIg5gDCaAaQ0B6mWz36YrXlmQSToNY'
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)