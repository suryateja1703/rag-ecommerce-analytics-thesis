import os
import sys
import time
import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

load_dotenv()
# Use Streamlit secrets if available
import streamlit as st
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key, default)
# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="RAG Analytics — Claude Opus 4.6",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        import chromadb
        import anthropic
        from sentence_transformers import SentenceTransformer

        embed_model   = SentenceTransformer(
            "BAAI/bge-m3",
            local_files_only=True
        )
        claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        chroma_client = chromadb.PersistentClient(
            path=st.secrets.get("CHROMA_PERSIST_PATH", os.getenv("CHROMA_PERSIST_PATH", "./embeddings"))
        )
        collection = chroma_client.get_collection("amazon_products")
        return embed_model, claude_client, collection, True
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, False

# ── Search & Generate ─────────────────────────────
def search_products(query, embed_model, collection, k=5):
    query_embedding = embed_model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def generate_insight(query, results, claude_client):
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join([
        f"Product {i+1}:\n"
        f"Title: {m.get('title','N/A')}\n"
        f"Brand: {m.get('store','N/A')}\n"
        f"Price: ${m.get('price','N/A')}\n"
        f"Rating: {m.get('average_rating','N/A')}/5.0\n"
        f"Category: {m.get('categories','N/A')}\n"
        f"Details: {d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    ])

    response = claude_client.messages.create(
        model=os.getenv("PRIMARY_MODEL", "claude-opus-4-6"),
        max_tokens=1024,
        system="""You are an expert e-commerce business analyst.
Analyze the provided Amazon Electronics product data and generate
structured, actionable business insights. Be specific and data-driven.""",
        messages=[{
            "role": "user",
            "content": f"Products:\n{context}\n\nQuestion: {query}\n\nProvide a structured analytical insight."
        }]
    )
    return response.content[0].text, docs, metas

# ── Sidebar ───────────────────────────────────────
def render_sidebar(collection):
    st.sidebar.markdown("## ⚙️ Settings")
    k = st.sidebar.slider("Products to retrieve", 3, 10, 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Database Stats")
    if collection:
        st.sidebar.metric("Total Vectors", f"{collection.count():,}")
        st.sidebar.metric("LLM Model", "Claude Opus 4.6")
        st.sidebar.metric("Embeddings", "BGE-M3")
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **UMass Dartmouth**
    MS Data Science Thesis
    2026
    """)
    return k

# ── Main ──────────────────────────────────────────
def main():
    st.markdown(
        '<div class="main-header">🤖 RAG-Powered E-Commerce Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Claude Opus 4.6 + BGE-M3 + ChromaDB | UMass Dartmouth Thesis 2026</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading models..."):
        embed_model, claude_client, collection, loaded = load_resources()

    k = render_sidebar(collection)

    if not loaded:
        st.error("❌ System not ready! Run embedder.py first.")
        return

    st.success(f"✅ System ready! {collection.count():,} vectors loaded")

    # ── Tabs ──────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Query Interface",
        "📊 Analytics",
        "📈 Evaluation"
    ])

    # ── Tab 1 ─────────────────────────────────────
    with tab1:
        st.markdown("### 💬 Ask a Business Question")

        sample_queries = [
            "What are the best wireless headphones under $100?",
            "Which laptop accessories have the highest ratings?",
            "What are common features of highly rated cameras?",
            "Which brands dominate the smart home category?",
            "What are the most affordable 4K monitors?",
        ]

        query = st.text_input(
            "Enter your question:",
            placeholder="e.g. What are the best wireless headphones under $100?"
        )

        st.markdown("**💡 Sample Questions:**")
        cols = st.columns(3)
        for i, sample in enumerate(sample_queries[:3]):
            with cols[i]:
                if st.button(sample[:35] + "...", key=f"s{i}"):
                    query = sample

        if st.button("🔍 Generate Insight", type="primary") and query:
            with st.spinner("🔍 Searching products..."):
                t1      = time.time()
                results = search_products(query, embed_model, collection, k=k)
                t1      = time.time() - t1

            with st.spinner("🤖 Claude Opus 4.6 generating insight..."):
                t2                    = time.time()
                insight, docs, metas  = generate_insight(query, results, claude_client)
                t2                    = time.time() - t2

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retrieved", k)
            c2.metric("Search Time", f"{t1:.2f}s")
            c3.metric("Gen Time", f"{t2:.2f}s")
            c4.metric("Total Time", f"{t1+t2:.2f}s")

            # Insight
            st.markdown("### 💡 Generated Insight")
            st.markdown(
                f'<div class="insight-box">{insight}</div>',
                unsafe_allow_html=True
            )

            # Products
            st.markdown("### 📦 Retrieved Products")
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                with st.expander(f"Product {i+1}: {meta.get('title','N/A')[:50]}..."):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Brand", meta.get("store", "N/A"))
                    c2.metric("Price", f"${meta.get('price','N/A')}")
                    c3.metric("Rating", f"{meta.get('average_rating','N/A')}/5")
                    st.text(doc[:300] + "...")

    # ── Tab 2 ─────────────────────────────────────
    with tab2:
        st.markdown("### 📊 Product Analytics")
        chunks_file = "./data/processed/processed_chunks.csv"
        if Path(chunks_file).exists():
            with st.spinner("Loading data..."):
                df = pd.read_csv(chunks_file, nrows=50000)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Chunks", f"{len(df):,}")
            c2.metric("Unique Products", f"{df['product_id'].nunique():,}")
            c3.metric("Avg Rating", f"{pd.to_numeric(df['average_rating'], errors='coerce').mean():.2f}")
            c4.metric("Categories", f"{df['main_category'].nunique():,}")

            col1, col2 = st.columns(2)
            with col1:
                df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
                fig = px.histogram(
                    df, x="average_rating",
                    title="⭐ Rating Distribution",
                    color_discrete_sequence=["#1f77b4"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cat_counts = df["main_category"].value_counts().head(8)
                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title="📦 Product Categories"
                )
                st.plotly_chart(fig, use_container_width=True)

            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df_price    = df[df["price"].notna() & (df["price"] < 500)]
            if len(df_price) > 0:
                fig = px.histogram(
                    df_price, x="price", nbins=50,
                    title="💰 Price Distribution",
                    color_discrete_sequence=["#2ca02c"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run preprocessor.py first!")

    # ── Tab 3 ─────────────────────────────────────
    with tab3:
        st.markdown("### 📈 Evaluation Results")
        if Path("evaluation_results.csv").exists():
            results_df = pd.read_csv("evaluation_results.csv")
            st.dataframe(results_df, use_container_width=True)
            metrics = results_df.columns.tolist()
            values  = results_df.iloc[0].tolist()
            fig = px.bar(
                x=metrics, y=values,
                title="📊 All Evaluation Metrics",
                color=values,
                color_continuous_scale="Viridis",
                labels={"x": "Metric", "y": "Score"}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run metrics.py first to see evaluation results!")
            st.code("python src/evaluation/metrics.py")

if __name__ == "__main__":
    main()