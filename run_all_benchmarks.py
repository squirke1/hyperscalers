import subprocess
import sys

# The question to use for all benchmarks (edit as needed or pass via sys.argv)
question = "I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?"

# Paths to your scripts
azure_script = "azure_openai_demo.py"
gcp_script = "gcp_vertexai_demo.py"
aws_script = "aws_bedrock_claude_demo.py"

# List of scripts to run
scripts = [
    ("Azure OpenAI", azure_script),
    ("GCP Vertex AI", gcp_script),
    ("AWS Bedrock Claude", aws_script),
]

for name, script in scripts:
    print(f"\n=== Running {name} Benchmark ===\n")
    try:
        subprocess.run(
            [sys.executable, script, "--question", question],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

print("\nAll benchmarks completed.")