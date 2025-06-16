# Hyperscalers Benchmarking

This repository contains a practical benchmarking toolkit for comparing the performance and output of large language models (LLMs) from the three major cloud providers: **Azure OpenAI (Llama 3)**, **Google Vertex AI (Llama 3)**, and **AWS Bedrock (Llama 3)**. The suite is designed to help developers, architects, and decision-makers evaluate these services side by side using real API calls and consistent prompts.

---

## What these scripts do

- **Runs the same prompt** across Azure, Google, and AWS Llama 3 APIs.
- **Measures and records** response time, token usage, character count, word count, and estimated cost.
- **Exports results** and averages to CSV files for each provider.
- **Aggregates and summarizes** results for easy comparison.
- **Lets you customize** the prompt and output file via command-line arguments.

---

## Requirements

- Python 3.8 or newer
- API access and credentials for:
    - Azure OpenAI (Llama 3 deployment)
    - Google Vertex AI (Llama 3 model)
    - AWS Bedrock (Llama 3 model)
- Python packages: `boto3`, `google-genai`, `vertexai`, `azure-ai-inference`, `tiktoken`

---

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/squirke1/hyperscalers
cd hyperscalers
```

### 2. Set Environment Variables

#### **Azure OpenAI**
```sh
export AZURE_LLAMAC3_API_KEY="your-azure-api-key"
export AZURE_LLAMAC3_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
export AZURE_LLAMAC3_MODEL_NAME="your-azure-deployment-name"
```

#### **Google Vertex AI**
```sh
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# Make sure you are authenticated with Google Cloud SDK and have access to Vertex AI
```

#### **AWS Bedrock**
```sh
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_DEFAULT_REGION="us-east-2"
# Ensure your IAM user/role has Bedrock invoke permissions and model access
```

### 3. Install Python Dependencies

```sh
pip install boto3 google-genai vertexai azure-ai-inference tiktoken
```

---

## Usage

### Run All Benchmarks

```sh
python run_all_benchmarks.py
```

- By default, this will use the standard comparison prompt.
- To use a custom question, edit the `question` variable in `run_all_benchmarks.py` or modify the script to accept a command-line argument.

### Run a Single Provider

Each provider script can be run individually:

```sh
python azure_llama_demo.py --question "custom question" --csv "azure_llama_results.csv"
python gcp_llama_demo.py --question "custom question" --csv "gcp_llama_results.csv"
python aws_llama_demo.py --question "custom question" --csv "aws_llama_results.csv"
```

---

## Output

- Each script writes detailed results and averages to its own CSV file (e.g., `azure_llama_results.csv`, `gcp_llama_results.csv`, `aws_llama_results.csv`).
- After all scripts run, `run_all_benchmarks.py` creates:
  - `benchmark_summary.csv` — Averages from each provider (providers as rows).
  - `benchmark_summary_transposed.csv` — Averages from each provider (metrics as rows, providers as columns) for easy comparison.

---

## Metrics Collected

For each run and in the averages, the following metrics are recorded:

- Response Time (seconds)
- Prompt Tokens
- Completion Tokens
- Total Tokens
- Characters
- Words
- Estimated Cost (USD)

---

## Example Summary Table

| Metric                   | Azure Llama 3 | GCP Llama 3 | AWS Bedrock Llama 3 |
|--------------------------|---------------|-------------|---------------------|
| Avg. Response Time (s)   | ...           | ...         | ...                 |
| Avg. Prompt Tokens       | ...           | ...         | ...                 |
| Avg. Completion Tokens   | ...           | ...         | ...                 |
| Avg. Total Tokens        | ...           | ...         | ...                 |
| Avg. Characters          | ...           | ...         | ...                 |
| Avg. Words               | ...           | ...         | ...                 |
| Avg. Cost                | ...           | ...         | ...                 |

---

## Notes

- Ensure you have access to the required models and regions in each cloud provider.
- API usage may incur costs—monitor your usage in each provider's console.
- For best results, use the same or similar prompt for all providers.
- **Azure:** If you experience slowness in `eastus`, try deploying your resource in another region such as `westus` or `swedencentral`.
- **Google Vertex AI:** Llama 3 models are typically available in `us-central1`.
- **AWS Bedrock:** Make sure your IAM user/role has access to the Llama 3 model and is using a supported region (e.g., `us-east-2`).

---

## License

MIT License

---

## Contact

For questions or contributions, please open an issue or submit a pull request.
