import os
import time
from openai import AzureOpenAI

# Use environment variables for sensitive data
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Use the base endpoint

if endpoint is None:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

model_name = "gpt-4"
deployment = "gpt-4.1-sq"

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

num_runs = 5  # Number of times to call the API

response_times = []
prompt_tokens_list = []
completion_tokens_list = []
total_tokens_list = []

for i in range(num_runs):
    start_time = time.time()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Hello.",
            },
            {
                "role": "user",
                "content": "I want to compare hyperscalers to see which is the best option for enterprise?",
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )
    end_time = time.time()
    elapsed = end_time - start_time
    response_times.append(elapsed)

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens_list.append(getattr(usage, "prompt_tokens", 0))
        completion_tokens_list.append(getattr(usage, "completion_tokens", 0))
        total_tokens_list.append(getattr(usage, "total_tokens", 0))
    else:
        prompt_tokens_list.append(0)
        completion_tokens_list.append(0)
        total_tokens_list.append(0)

    print(f"Run {i+1}:")
    print(response.choices[0].message.content)
    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {prompt_tokens_list[-1]}")
    print(f"Completion tokens: {completion_tokens_list[-1]}")
    print(f"Total tokens: {total_tokens_list[-1]}")
    print("-" * 40)

print("Averages over", num_runs, "runs:")
print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")

