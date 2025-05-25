import os
from openai import AzureOpenAI

# Validate environment variables before using them
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
if azure_endpoint is None:
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable is not set.")

# Now it's safe to use them
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2023-07-01-preview"
)

# Send a chat completion request
response = client.chat.completions.create(
    model="gpt-35-turbo-hp-test",  # <-- Deployment name, not model ID
    messages=[
        {"role": "user", "content": "Explain Retrieval-Augmented Generation (RAG)."}
    ]
)

# Print the response
print("Azure GPT-3.5 Response:\n", response.choices[0].message.content)
