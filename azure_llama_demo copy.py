import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load endpoint from environment variable
endpoint = os.getenv("AZURE_LLAMAC3_ENDPOINT")
if endpoint is None:
    raise ValueError("AZURE_LLAMAC3_ENDPOINT environment variable is not set.")

model_name = os.getenv("AZURE_LLAMAC3_MODEL_NAME")
if model_name is None:
    raise ValueError("AZURE_LLAMAC3_MODEL_NAME environment variable is not set.")

# Load API key from environment variable
api_key = os.getenv("AZURE_LLAMAC3_API_KEY")
if api_key is None:
    raise ValueError("AZURE_LLAMAC3_API_KEY environment variable is not set.")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?"),
    ],
    max_tokens=2048,
    temperature=0.8,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    model=model_name
)

print(response.choices[0].message.content)