# gcp_vertexai_demo.py
# This script benchmarks Google Vertex AI Gemini model responses, including timing, token usage, and output statistics.

import os
import time
import csv
import argparse
from google import genai
from google.genai import types
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part


# Parse command-line arguments for the question and CSV filename
parser = argparse.ArgumentParser(description="Benchmark GCP Vertex AI Llama model responses.")
parser.add_argument(
    "--question",
    type=str,
    default="I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?",
    help="The question to send to the Llama model."
)
parser.add_argument(
    "--csv",
    type=str,
    default="gcp_llama_results.csv",
    help="The CSV filename to write results to."
)
args = parser.parse_args()
prompt = args.question
csv_filename = args.csv

def generate():
    # Get the GCP project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

    # Initialize the Vertex AI client for Llama models
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel("llama-3.3-70b-instruct-maas")  # Model name to use

    # Use the prompt from the command line
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        ),
    ]

    # Configure generation parameters and safety settings
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "max_output_tokens": 3000,  # Allow enough tokens for 600+ words
    }

    num_runs = 5  # Number of times to call the API for benchmarking

    # Llama 3.3 70B pricing (USD per 1000 tokens)
    input_price_per_1k_tokens = 0.00072  # $0.72 / million tokens = $0.00072 / 1k tokens
    output_price_per_1k_tokens = 0.00072  # $0.72 / million tokens = $0.00072 / 1k tokens

    # Lists to store metrics for each run
    response_times = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    responses = []
    costs = []

    # Run the API call multiple times to gather statistics
    for i in range(num_runs):
        start_time = time.time()  # Start timing
        # Generate content using the Llama model
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=False  # Disable streaming for simpler token counting
        )
        end_time = time.time()  # End timing
        elapsed = end_time - start_time
        response_times.append(elapsed)

        # Extract the response text
        full_response = response.text
        responses.append(full_response)

        # Extract token usage information
        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        total_tokens = getattr(response.usage_metadata, "total_token_count", 0)

        # Completion tokens are not directly available, so calculate them
        completion_tokens = total_tokens - prompt_tokens

        # Calculate the cost for this run
        input_cost = (prompt_tokens / 1000) * input_price_per_1k_tokens
        output_cost = (completion_tokens / 1000) * output_price_per_1k_tokens
        total_cost = input_cost + output_cost
        costs.append(total_cost)

        # Store token usage for this run
        prompt_tokens_list.append(prompt_tokens)
        completion_tokens_list.append(completion_tokens)
        total_tokens_list.append(total_tokens)

        # Count characters and words in the response
        char_count = len(full_response)
        word_count = len(full_response.split())

        # Print metrics for this run
        print(f"Run {i+1}:")
        print(full_response)
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Characters: {char_count}")
        print(f"Words: {word_count}")
        print(f"Estimated cost (USD): {total_cost:.6f}")
        print("-" * 40)

    # Print averages for all runs
    print("Averages over", num_runs, "runs:")
    print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
    print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
    print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
    print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")
    print(f"Average cost: {sum(costs)/num_runs:.6f}")

    # Write all results to a CSV file for later analysis
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow([
            "Run", "Response Time (s)", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Characters", "Words", "Cost (USD)", "Response"
        ])
        # Write each run's data
        for i in range(num_runs):
            resp_text = responses[i]
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

# Run the benchmarking function
if __name__ == "__main__":
    generate()