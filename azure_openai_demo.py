# azure_openai_demo.py

import openai

openai.api_type = "azure"
openai.api_base = "https://<your-resource-name>.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "<your-azure-api-key>"

response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages=[
        {"role": "user", "content": "Explain Retrieval-Augmented Generation (RAG)."}
    ]
)

print("Azure GPT-3.5 Response:\n", response["choices"][0]["message"]["content"])