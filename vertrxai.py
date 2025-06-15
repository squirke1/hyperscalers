import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "smart-impact-459512-f3"
LOCATION = "us-central1"
# Use the direct model ID for the fully-managed Llama model
# Example: "llama-3-8b-instruct" or "llama-3-70b-instruct"
MODEL_ID = "llama-3.3-70b-instruct-maas" # <--- CHANGED THIS LINE

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- Load the model ---
model = GenerativeModel(MODEL_ID)

# --- Define the prompt ---
prompt = "Tell me a short, funny story about a talking cat who tries to become a famous chef."

# --- Generate content ---
print(f"Sending prompt to {MODEL_ID} in {LOCATION}...")
response = model.generate_content(
    [Part.from_text(prompt)],
    generation_config={
        "max_output_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    safety_settings=[]
)

# --- Print the response ---
if response.candidates:
    for candidate in response.candidates:
        print("\n--- Response ---")
        print(candidate.text)
else:
    print("No candidates found in the response.")
    if response.prompt_feedback:
        print(f"Prompt feedback: {response.prompt_feedback}")