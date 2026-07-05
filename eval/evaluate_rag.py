import os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
)

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT", "https://docuassist-search.search.windows.net")
SEARCH_KEY = os.getenv("SEARCH_KEY")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "https://eastus.api.cognitive.microsoft.com/")
AOAI_KEY = os.getenv("AOAI_KEY")
INDEX_NAME = "docuassist-index"

model_config = {
    "azure_endpoint": AOAI_ENDPOINT,
    "api_key": AOAI_KEY,
    "azure_deployment": "gpt-5-mini",
    "api_version": "2024-06-01",
}

test_questions = [
    "What is MLOps and why is it important?",
    "How does drift detection work in production ML systems?",
    "What is RAG and how does it use Azure AI Search?",
    "How does Responsible AI content filtering work in Azure?",
    "What tools can I use for model versioning?",
]

oai = AzureOpenAI(api_key=AOAI_KEY, api_version="2024-06-01", azure_endpoint=AOAI_ENDPOINT)
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

SYSTEM_PROMPT = (
    "You are DocuAssist, a helpful assistant that answers questions "
    "based on the provided context.\nRules:\n"
    "- Only use information from the context below to answer.\n"
    "- If the context doesn't contain the answer, say "
    "\"I don't have enough information to answer that.\"\n"
    "- Cite which document(s) you used by mentioning their title."
)

def run_rag(question):
    vec = oai.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding
    results = search_client.search(
        search_text=question,
        vector_queries=[VectorizedQuery(vector=vec, k_nearest_neighbors=3, fields="contentVector")],
        select=["title", "content"], top=3,
    )
    chunks = [f"[{r['title']}]: {r['content']}" for r in results]
    context = "\n\n".join(chunks)
    resp = oai.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=1500,
    )
    return resp.choices[0].message.content, context

print("Running RAG pipeline on test questions...\n")
eval_data = []
for q in test_questions:
    answer, context = run_rag(q)
    eval_data.append({"query": q, "response": answer, "context": context})
    print(f"Q: {q}")
    print(f"A: {answer[:120]}...\n")

print("=" * 60)
print("Running evaluators (LLM-as-judge)...\n")

g_eval = GroundednessEvaluator(model_config=model_config, is_reasoning_model=True)
r_eval = RelevanceEvaluator(model_config=model_config, is_reasoning_model=True)
c_eval = CoherenceEvaluator(model_config=model_config, is_reasoning_model=True)

all_scores = {"groundedness": [], "relevance": [], "coherence": []}

for i, item in enumerate(eval_data):
    print(f"Evaluating Q{i+1}: {item['query'][:50]}...")
    g = g_eval(query=item["query"], response=item["response"], context=item["context"])
    r = r_eval(query=item["query"], response=item["response"])
    c = c_eval(query=item["query"], response=item["response"])
    gs = g.get("groundedness", g.get("gpt_groundedness", "N/A"))
    rs = r.get("relevance", r.get("gpt_relevance", "N/A"))
    cs = c.get("coherence", c.get("gpt_coherence", "N/A"))
    all_scores["groundedness"].append(float(gs) if gs != "N/A" else 0)
    all_scores["relevance"].append(float(rs) if rs != "N/A" else 0)
    all_scores["coherence"].append(float(cs) if cs != "N/A" else 0)
    print(f"  Groundedness: {gs}  Relevance: {rs}  Coherence: {cs}")

print("\n" + "=" * 60)
print("EVALUATION SUMMARY (1-5 scale)")
print("=" * 60)
for metric, scores in all_scores.items():
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  {metric:15s}: avg={avg:.2f}  min={min(scores):.1f}  max={max(scores):.1f}")
print(f"\nTotal test questions: {len(test_questions)}")
print("Pass threshold: Groundedness >= 4.0")
passed = all(s >= 4.0 for s in all_scores["groundedness"])
print(f"Result: {'PASS' if passed else 'FAIL'}")
