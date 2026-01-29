import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""
I use this utility script to verify the API connection and environment 
configuration. It ensures that the OpenRouter credentials and the 
Claude 3.5 Sonnet model are properly authenticated and reachable 
before initiating the production-scale generation tasks.
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

try:
    response = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[{"role": "user", "content": "Say Hello, World!"}],
    )
    print("Connection Successful:", response.choices[0].message.content)
except Exception as e:
    print("Connection Failed:", e)
