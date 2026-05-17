import os
import time
import anthropic
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import mlflow
import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PERSIST_PATH", "./embeddings")
PRIMARY_MODEL   = os.getenv("PRIMARY_MODEL", "claude-opus-4-6")
EMBEDDING_MODEL = "BAAI/bge-m3"
RETRIEVAL_K     = int(os.getenv("RETRIEVAL_K", 5))
TEMPERATURE     = float(os.getenv("TEMPERATURE", 0.3))

SYSTEM_PROMPT = """You are an expert business intelligence analyst
specializing in e-commerce product analytics.
You will be provided with Amazon Electronics product metadata
retrieved from a large product database.
Your task is to:
1. Analyze the product information carefully
2. Identify key patterns, themes, and insights
3. Quantify observations where evidence supports it
4. Present findings as a structured, executive-ready analytical insight
5. Ground ALL claims in the provided product data
6. End with a clear business recommendation
Be specific, data-driven, and actionable."""

def load_models():
    print("🤖 Loading models...")
    embed_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True
    )
    claude_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    print("✅ All models loaded!")
    return embed_model, claude_client

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("amazon_products")

def retrieve_chunks(query, embed_model, collection, k=20):
    query_embedding = embed_model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text"    : doc,
            "metadata": meta,
            "distance": dist
        })
    return chunks

def rerank_chunks(chunks, top_k=5):
    sorted_chunks = sorted(chunks, key=lambda x: x["distance"])
    return sorted_chunks[:top_k]

def generate_insight(query, chunks, claude_client):
    context = "\n\n".join([
        f"Product {i+1}:\n"
        f"Title: {c['metadata'].get('title', 'N/A')}\n"
        f"Brand: {c['metadata'].get('store', 'N/A')}\n"
        f"Category: {c['metadata'].get('categories', 'N/A')}\n"
        f"Price: ${c['metadata'].get('price', 'N/A')}\n"
        f"Rating: {c['metadata'].get('average_rating', 'N/A')}/5.0\n"
        f"Details: {c['text']}"
        for i, c in enumerate(chunks)
    ])
    user_message = f"""Based on these Amazon Electronics products:

{context}

Business Question: {query}

Please provide a structured analytical insight with:
1. Key Finding
2. Supporting Evidence
3. Business Recommendation"""

    response = claude_client.messages.create(
        model=PRIMARY_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

def run_rag_pipeline(query, embed_model, claude_client, collection):
    with mlflow.start_run(nested=True):
        mlflow.log_param("query", query[:100])
        mlflow.log_param("model", PRIMARY_MODEL)
        mlflow.log_param("retrieval_k", RETRIEVAL_K)

        start         = time.time()
        chunks        = retrieve_chunks(query, embed_model, collection, k=20)
        retrieve_time = time.time() - start

        chunks = rerank_chunks(chunks, top_k=RETRIEVAL_K)

        start         = time.time()
        insight       = generate_insight(query, chunks, claude_client)
        generate_time = time.time() - start

        mlflow.log_metric("retrieve_time_sec", round(retrieve_time, 3))
        mlflow.log_metric("generate_time_sec", round(generate_time, 3))
        mlflow.log_metric("total_time_sec", round(retrieve_time + generate_time, 3))

    return insight, chunks

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG Pipeline Test — Claude Opus 4.6")
    print("=" * 60)

    mlflow.set_experiment("rag_pipeline_test")
    embed_model, claude_client = load_models()

    try:
        collection = get_collection()
        print(f"✅ ChromaDB loaded — {collection.count():,} vectors")
    except Exception as e:
        print(f"❌ ChromaDB not ready: {e}")
        exit()

    test_queries = [
        "What are the best wireless headphones under $100?",
        "Which laptop accessories have the highest ratings?",
        "What are common features of highly rated cameras?",
    ]

    with mlflow.start_run(run_name="pipeline_test"):
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("⏳ Running RAG pipeline...")
            insight, chunks = run_rag_pipeline(
                query, embed_model, claude_client, collection
            )
            print("\n💡 Insight:")
            print("-" * 40)
            print(insight)
            print("-" * 40)

    print("\n✅ RAG Pipeline Test Complete!")