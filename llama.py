import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

api_key = os.getenv("AZURE_OPENAI_API_KEY")
if api_key is None:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

model_name = "your-model-name"  # Replace with your actual model name

client = ChatCompletionsClient(
    endpoint="https://steph-mb47rkot-eastus2.services.ai.azure.com/",
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?"),
    ],
    max_tokens=2048,
    temperature=0.8,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    model=model_name
)

print(response.choices[0].message.content)