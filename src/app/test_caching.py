import os, time
from openai import AzureOpenAI

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "https://eastus.api.cognitive.microsoft.com/")
AOAI_KEY = os.getenv("AOAI_KEY")

oai = AzureOpenAI(api_key=AOAI_KEY, api_version="2024-06-01", azure_endpoint=AOAI_ENDPOINT)

# Build a long static prefix (>1024 tokens) to trigger caching
# In production this would be your system prompt + tool definitions + static context
long_context = """You are DocuAssist, an enterprise RAG assistant for MLOps documentation.

INSTRUCTIONS:
- Answer only from the provided context
- Cite document titles
- Be concise but thorough
- If unsure, say so

KNOWLEDGE BASE:
""" + "\n".join([
    f"Document {i}: " + "This is a detailed technical document about MLOps practices. " * 20
    for i in range(1, 21)
])

# Two different user questions with the SAME long prefix
questions = [
    "What is model drift and how do I detect it?",
    "How does CI/CD work for machine learning?",
    "What are the best practices for model versioning?",
]

print(f"Prefix length estimate: ~{len(long_context.split()) * 1.3:.0f} tokens\n")

for i, q in enumerate(questions):
    start = time.time()
    response = oai.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": long_context},
            {"role": "user", "content": q},
        ],
        max_completion_tokens=300,
    )
    elapsed = time.time() - start
    usage = response.usage

    cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
    total_prompt = usage.prompt_tokens

    print(f"Q{i+1}: {q[:50]}...")
    print(f"  Prompt tokens: {total_prompt} | Cached: {cached} ({cached/total_prompt*100:.0f}%)")
    print(f"  Completion tokens: {usage.completion_tokens}")
    print(f"  Latency: {elapsed:.2f}s\n")

    time.sleep(2)  # small gap between requests

print("If cached > 0 on Q2/Q3, prompt caching is working.")
print("Cached tokens are billed at 50% discount on Standard deployments.")
