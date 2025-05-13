# aws_bedrock_claude_demo.py

import boto3
import json
import time

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = {
    "prompt": "\n\nHuman: Explain the concept of Retrieval-Augmented Generation (RAG) in AI.\n\nAssistant:",
    "max_tokens_to_sample": 300,
    "temperature": 0.7,
    "stop_sequences": ["\n\nHuman:"]
}

start = time.time()
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(prompt)
)
end = time.time()

output = response["body"].read().decode()
print(f"Claude 3 Sonnet Response (Latency: {round(end - start, 2)}s):\n", output)