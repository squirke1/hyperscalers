import os
from openai import AzureOpenAI

# Validate environment variables before using them
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
if azure_endpoint is None:
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable is not set.")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=api_key,
    azure_endpoint=azure_endpoint
)

model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo-hp-test"

response = client.chat.completions.create(
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
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)



"""
# Send a chat completion request

response = client.chat.completions.create(
    model="gpt-35-turbo-hp-test",  # <-- Deployment name, not model ID
    messages=[
        {"role": "user", "content": "Explain Retrieval-Augmented Generation (RAG)."}
    ]
)

# Print the response
print("Azure GPT-3.5 Response:\n", response.choices[0].message.content)
"""
