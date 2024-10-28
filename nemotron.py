from openai import OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi--B2EmhDjkwTP874m-7HKS"
)
try:
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2
    )
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")