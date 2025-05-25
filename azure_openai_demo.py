import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import cast
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Read from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = cast(str, os.getenv("AZURE_OPENAI_ENDPOINT"))
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
deployment_name = "o4-mini-hp-test"

# Ensure no variables are missing
if not all([api_key, endpoint, deployment_name]):
    raise ValueError("Missing required environment variables.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=500
)

# Output
print(response.choices[0].message.content)
logging.info(response.choices[0].message.content)
