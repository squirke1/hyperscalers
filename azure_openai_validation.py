import os
from openai import AzureOpenAI
from openai import OpenAIError

def validate_env_vars():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise EnvironmentError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT environment variable.")

    return api_key, endpoint

def test_deployment(client: AzureOpenAI, deployment_name: str):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"✅ Deployment '{deployment_name}' is valid.")
        print("Sample response:", response.choices[0].message.content)
    except OpenAIError as e:
        print(f"❌ Deployment '{deployment_name}' failed. Reason:\n{e}")

def main():
    api_key, endpoint = validate_env_vars()

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-07-01-preview",
        azure_endpoint=endpoint
    )

    # Replace with your actual Azure OpenAI deployment name
    deployment_name = "gpt-35-turbo"
    test_deployment(client, deployment_name)

if __name__ == "__main__":
    main()
