import os
import time
import csv
import argparse
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Parse command-line arguments for the question and CSV filename
parser = argparse.ArgumentParser(description="Benchmark Azure Llama 3 model responses.")
parser.add_argument(
    "--question",
    type=str,
    default="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?",
    help="The question to send to the Azure Llama 3 model."
)
parser.add_argument(
    "--csv",
    type=str,
    default="llama3_results.csv",
    help="The CSV filename to write results to."
)
args = parser.parse_args()
prompt = args.question
csv_filename = args.csv
model_name = "Llama-3-70B-Instruct"  # Model name for Azure Llama 3

# Load sensitive data from environment variables
api_key = os.getenv("AZURE_LLAMAC3_API_KEY")
if api_key is None:
    raise ValueError("AZURE_LLAMAC3_API_KEY environment variable is not set.")

endpoint = os.getenv("AZURE_LLAMAC3_ENDPOINT")
# Azure Llama endpoint

if endpoint is None:
    raise ValueError("AZURE_LLAMAC3_ENDPOINT environment variable is not set.")

deployment = os.getenv("AZURE_LLAMAC3_DEPLOYMENT")  # Deployment name from environment variable

if deployment is None:
    raise ValueError("AZURE_LLAMAC3_DEPLOYMENT environment variable is not set.")

# Initialize the Azure Llama client
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

num_runs = 5  # Number of times to call the API for benchmarking

# Lists to store metrics for each run
response_times = []
prompt_tokens_list = []
completion_tokens_list = []
total_tokens_list = []
responses = []
costs = []

# Pricing for Llama-3-70B-Instruct (as of June 2024, pay-as-you-go)
input_token_price = 0.00071  # USD per 1K input tokens
output_token_price = 0.00071  # USD per 1K output tokens

# Run the API call multiple times to gather statistics
for i in range(num_runs):
    start_time = time.time()  # Start timing
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=prompt),  # Use the prompt from the command line
        ],
        max_tokens=2048,
        temperature=0.8,
        top_p=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=model_name
    )
    end_time = time.time()  # End timing
    elapsed = end_time - start_time
    response_times.append(elapsed)

    # Extract token usage information if available
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

    # Store token usage for this run
    prompt_tokens_list.append(prompt_tokens)
    completion_tokens_list.append(completion_tokens)
    total_tokens_list.append(total_tokens)

    # Calculate cost for this call
    input_cost = (prompt_tokens / 1000) * input_token_price
    output_cost = (completion_tokens / 1000) * output_token_price
    total_cost = input_cost + output_cost
    costs.append(total_cost)

    # Get the response text
    resp_text = response.choices[0].message.content or ""
    responses.append(resp_text)

    # Count characters and words in the response
    char_count = len(resp_text)
    word_count = len(resp_text.split())

    # Print metrics for this run
    print(f"Run {i+1}:")
    print(resp_text)
    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Characters: {char_count}")
    print(f"Words: {word_count}")
    print(f"Estimated cost (USD): {total_cost:.6f}")
    print("-" * 40)

# Write all results to a CSV file for later analysis
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow([
        "Run", "Response Time (s)", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Characters", "Words", "Cost (USD)", "Response"
    ])
    # Write each run's data
    for i in range(num_runs):
        resp_text = responses[i] or ""
        char_count = len(resp_text)
        word_count = len(resp_text.split())
        writer.writerow([
            i + 1,
            f"{response_times[i]:.2f}",
            prompt_tokens_list[i],
            completion_tokens_list[i],
            total_tokens_list[i],
            char_count,
            word_count,
            f"{costs[i]:.6f}",
            resp_text.replace('\n', ' ')
        ])
    # Write averages row
    writer.writerow([])
    writer.writerow([
        "Average",
        f"{sum(response_times)/num_runs:.2f}",
        f"{sum(prompt_tokens_list)/num_runs:.2f}",
        f"{sum(completion_tokens_list)/num_runs:.2f}",
        f"{sum(total_tokens_list)/num_runs:.2f}",
        f"{sum(len(r) for r in responses)/num_runs:.2f}",
        f"{sum(len(r.split()) for r in responses)/num_runs:.2f}",
        f"{sum(costs)/num_runs:.6f}",
        ""
    ])

print(f"Results written to {csv_filename}")

# Print averages to the console for quick reference
print("Averages over", num_runs, "runs:")
print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")
print(f"Average characters: {sum(len(r) for r in responses)/num_runs:.2f}")
print(f"Average words: {sum(len(r.split()) for r in responses)/num_runs:.2f}")