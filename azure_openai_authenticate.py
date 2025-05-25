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