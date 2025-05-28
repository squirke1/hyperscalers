import os
from openai import AzureOpenAI

# Use environment variables for sensitive data
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = "https://steph-mb47rkot-eastus2.openai.azure.com/"  # Use the base endpoint

model_name = "gpt-4"
deployment = "gpt-4.1-sq"


client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

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

print("Azure OpenAI client initialized successfully.")

