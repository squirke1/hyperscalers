# gcp_vertexai_demo.py

import os
from google import genai
from google.genai import types
import base64
import csv

def generate():
    import time

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="global",
    )

    model = "gemini-2.5-flash-preview-05-20"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""I want to compare hyperscalers to see which is the best option for enterprise?""")
            ]
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            )
        ],
    )

    num_runs = 5
    response_times = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    responses = []

    for i in range(num_runs):
        start_time = time.time()
        response_chunks = list(client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ))
        end_time = time.time()
        elapsed = end_time - start_time
        response_times.append(elapsed)

        # Concatenate all chunk texts
        full_response = "".join(chunk.text for chunk in response_chunks if hasattr(chunk, "text") and isinstance(chunk.text, str) and chunk.text is not None)
        responses.append(full_response)

        # Token usage (if available)
        usage = getattr(response_chunks[-1], "usage_metadata", None) if response_chunks else None
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0

        prompt_tokens_list.append(prompt_tokens)
        completion_tokens_list.append(completion_tokens)
        total_tokens_list.append(total_tokens)

        print(f"Run {i+1}:")
        print(full_response)
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print("-" * 40)

    print("Averages over", num_runs, "runs:")
    print(f"Average response time: {sum(response_times)/num_runs:.2f} seconds")
    print(f"Average prompt tokens: {sum(prompt_tokens_list)/num_runs:.2f}")
    print(f"Average completion tokens: {sum(completion_tokens_list)/num_runs:.2f}")
    print(f"Average total tokens: {sum(total_tokens_list)/num_runs:.2f}")

    # Write results to CSV
    csv_filename = "vertexai_results.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Run", "Response Time (s)", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Response"
        ])
        for i in range(num_runs):
            writer.writerow([
                i + 1,
                f"{response_times[i]:.2f}",
                prompt_tokens_list[i],
                completion_tokens_list[i],
                total_tokens_list[i],
                responses[i].replace('\n', ' ')
            ])
        # Write averages
        writer.writerow([])
        writer.writerow([
            "Average",
            f"{sum(response_times)/num_runs:.2f}",
            f"{sum(prompt_tokens_list)/num_runs:.2f}",
            f"{sum(completion_tokens_list)/num_runs:.2f}",
            f"{sum(total_tokens_list)/num_runs:.2f}",
            ""
        ])

    print(f"Results written to {csv_filename}")

generate()