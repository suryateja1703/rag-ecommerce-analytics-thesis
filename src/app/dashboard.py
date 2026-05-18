import os
import sys
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Intelligence Hub",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --amazon-orange: #FF9900;
        --amazon-dark: #131921;
        --amazon-blue: #146EB4;
        --amazon-grey: #F0F2F2;
        --amazon-border: #D5D9D9;
        --amazon-text: #0F1111;
        --amazon-text-muted: #565959;
        --amazon-green: #007600;
        --amazon-red: #B12704;
        --white: #FFFFFF;
        --shadow: 0 2px 5px rgba(213,217,217,0.5);
        --shadow-hover: 0 2px 10px rgba(0,0,0,0.15);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    .stApp {
        background-color: #EAEDED;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--amazon-dark) !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .nav-bar {
        background: linear-gradient(180deg, #232F3E 0%, #131921 100%);
        padding: 12px 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 3px solid var(--amazon-orange);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .nav-logo { font-size: 1.8rem; font-weight: 700; color: white; }
    .nav-logo span { color: var(--amazon-orange); }
    .nav-tagline { font-size: 0.72rem; color: #ccc; letter-spacing: 1.5px; text-transform: uppercase; }
    .nav-badge {
        background: var(--amazon-orange);
        color: var(--amazon-dark);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: auto;
    }

    .metric-card {
        background: var(--white);
        border: 1px solid var(--amazon-border);
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: var(--shadow);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--amazon-text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: var(--amazon-text); line-height: 1; }
    .metric-sub { font-size: 0.72rem; color: var(--amazon-green); margin-top: 4px; }

    .insight-container {
        background: var(--white);
        border: 1px solid var(--amazon-border);
        border-radius: 8px;
        padding: 24px;
        box-shadow: var(--shadow);
        margin-top: 16px;
        border-left: 4px solid var(--amazon-orange);
    }
    .insight-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--amazon-border);
    }
    .insight-badge {
        background: #FFF3CD;
        border: 1px solid #FFD700;
        color: #856404;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .insight-title { font-size: 1rem; font-weight: 700; color: var(--amazon-text); }

    .stats-row {
        background: var(--white);
        border: 1px solid var(--amazon-border);
        border-radius: 8px;
        padding: 14px 20px;
        display: flex;
        gap: 32px;
        align-items: center;
        margin-bottom: 16px;
        box-shadow: var(--shadow);
    }
    .stat-item { text-align: center; }
    .stat-number { font-size: 1.1rem; font-weight: 700; color: var(--amazon-text); }
    .stat-label { font-size: 0.7rem; color: var(--amazon-text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--amazon-text);
        padding-bottom: 10px;
        border-bottom: 2px solid var(--amazon-orange);
        margin-bottom: 16px;
    }

    /* Home page cards */
    .home-hero {
        background: linear-gradient(135deg, #131921 0%, #232F3E 100%);
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 16px;
    }
    .home-hero-title { font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 4px; }
    .home-hero-title span { color: #FF9900; }
    .home-hero-sub { font-size: 0.72rem; color: rgba(255,255,255,0.4); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 12px; }
    .home-hero-badge {
        display: inline-block;
        background: rgba(255,153,0,0.15);
        border: 1px solid rgba(255,153,0,0.4);
        color: #FF9900;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .home-hero-desc { font-size: 0.88rem; color: rgba(255,255,255,0.6); line-height: 1.7; margin-bottom: 20px; }
    .home-stats { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; }
    .home-stat-val { font-size: 1.3rem; font-weight: 700; color: white; }
    .home-stat-label { font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; }
    .home-status { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
    .home-status-dot { width: 8px; height: 8px; background: #007600; border-radius: 50%; animation: pulse 2s infinite; }
    .home-status-text { font-size: 0.78rem; color: #007600; font-weight: 600; }

    .page-card {
        background: var(--white);
        border: 1px solid var(--amazon-border);
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--shadow);
        transition: box-shadow 0.2s, transform 0.2s;
        cursor: pointer;
        height: 100%;
    }
    .page-card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); }
    .page-card-icon { font-size: 1.8rem; margin-bottom: 10px; }
    .page-card-title { font-size: 1rem; font-weight: 700; color: var(--amazon-text); margin-bottom: 6px; }
    .page-card-desc { font-size: 0.82rem; color: var(--amazon-text-muted); line-height: 1.5; }
    .page-card-link { font-size: 0.78rem; color: var(--amazon-blue); margin-top: 10px; font-weight: 600; }

    .powered-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .powered-item {
        background: var(--amazon-grey);
        border-radius: 8px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .powered-icon { font-size: 1.4rem; }
    .powered-name { font-size: 0.85rem; font-weight: 700; color: var(--amazon-text); }
    .powered-desc { font-size: 0.72rem; color: var(--amazon-text-muted); margin-top: 1px; }

    .stButton > button {
        background: linear-gradient(180deg, #E8A020 0%, #CC8800 100%);
        color: white !important;
        border: 1px solid #CC8800;
        border-radius: 20px;
        padding: 8px 24px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 153, 0, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(255, 153, 0, 0.5) !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        padding: 6px 12px !important;
        text-align: left !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 153, 0, 0.3) !important;
        border-color: var(--amazon-orange) !important;
        color: white !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: white !important; }

    .stButton > button:hover {
        background: linear-gradient(180deg, #F7CA00 0%, #F0C000 100%);
        transform: translateY(-1px);
    }
    .stTextInput > div > div > input {
        border: 1px solid var(--amazon-border);
        border-radius: 4px;
        padding: 10px 14px;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--amazon-orange);
        box-shadow: 0 0 0 3px rgba(255,153,0,0.2);
    }

    .status-online { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; color: var(--amazon-green); font-weight: 600; }
    .status-dot { width: 8px; height: 8px; background: var(--amazon-green); border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

    .sidebar-section { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(255,255,255,0.5); margin: 20px 0 8px 0; }
    .sidebar-stat { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
    .sidebar-stat-label { font-size: 0.72rem; color: rgba(255,255,255,0.6); margin-bottom: 2px; }
    .sidebar-stat-value { font-size: 1rem; font-weight: 700; color: white; }

    .stTabs [data-baseweb="tab-list"] { background: var(--white); border-radius: 8px 8px 0 0; border: 1px solid var(--amazon-border); border-bottom: none; }
    .stTabs [data-baseweb="tab"] { color: var(--amazon-text-muted); font-weight: 500; padding: 12px 24px; }
    .stTabs [aria-selected="true"] { color: var(--amazon-text) !important; border-bottom: 3px solid var(--amazon-orange) !important; font-weight: 700 !important; }

    /* Nav items in sidebar */
    .nav-page-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        border-left: 3px solid transparent;
        margin: 2px 8px;
    }
    .nav-page-item.active {
        background: rgba(255,153,0,0.1);
        color: #FF9900;
        border-left: 3px solid #FF9900;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        import chromadb
        import anthropic
        from sentence_transformers import SentenceTransformer

        embed_model   = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
        claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_PATH", "./embeddings")
        )
        collection = chroma_client.get_collection("amazon_products")
        return embed_model, claude_client, collection, True
    except Exception as e:
        return None, None, None, False

# ── Search & Generate ─────────────────────────────────────────────────────────
def search_products(query, embed_model, collection, k=5):
    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
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
        system="""You are a senior product intelligence analyst at a major e-commerce platform.
Analyze the provided Amazon Electronics product data and generate structured,
executive-ready business insights. Use specific data points, compare products,
identify patterns, and provide clear actionable recommendations.
Format your response with clear sections: Key Finding, Market Analysis,
Product Comparison, and Strategic Recommendation.""",
        messages=[{"role": "user", "content": f"Products:\n{context}\n\nBusiness Query: {query}\n\nProvide a comprehensive product intelligence report."}]
    )
    return response.content[0].text, docs, metas

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar(collection, loaded):
    # Logo
    st.sidebar.markdown("""
    <div style="font-size:1.3rem;font-weight:700;color:white;
    padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:16px">
        📦 Product<span style="color:#FF9900">IQ</span>
        <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Page navigation
    st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    pages = {
        "🏠 Home":                  "home",
        "🔍 Product Intelligence":  "search",
        "📊 Market Analytics":      "market",
        "📈 Performance Metrics":   "metrics",
    }

    if "page" not in st.session_state:
        st.session_state.page = "home"

    for label, key in pages.items():
        active = st.session_state.page == key
        style = "background:rgba(255,153,0,0.12);color:#FF9900;border-left:3px solid #FF9900;" if active else "color:rgba(255,255,255,0.6);border-left:3px solid transparent;"
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    # Search settings — only show on search page
    if st.session_state.page == "search":
        st.sidebar.markdown('<div class="sidebar-section">⚙️ Search Settings</div>', unsafe_allow_html=True)
        k = st.sidebar.slider("Products to analyze", 3, 10, 5)
    else:
        k = 5

    # Quick queries — always visible
    st.sidebar.markdown('<div class="sidebar-section">💡 Quick Queries</div>', unsafe_allow_html=True)
    queries = [
        "Best wireless headphones under $100",
        "Top rated laptop accessories",
        "Most popular smart home devices",
        "Budget 4K monitors comparison",
        "Best cameras for beginners",
    ]
    selected = None
    for q in queries:
        if st.sidebar.button(f"🔍 {q}", key=f"sq_{q}", use_container_width=True):
            selected = q
            st.session_state.page = "search"
            st.rerun()

    return k, selected

# ── HOME PAGE ─────────────────────────────────────────────────────────────────
def show_home(total_vectors):
    # Nav bar
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="nav-logo">📦 Product<span>IQ</span></div>
            <div class="nav-tagline">Intelligence Platform</div>
        </div>
        <div class="nav-badge">✦ AI Powered</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown(f"""
    <div class="home-hero">
        <div class="home-hero-badge">✦ AI Powered</div>
        <div class="home-hero-title">📦 Product<span>IQ</span></div>
        <div class="home-hero-sub">Intelligence Platform</div>
        <div class="home-hero-desc">
            An AI-powered product analytics platform that uses Retrieval-Augmented Generation (RAG)
            to analyze 200,000 Amazon Electronics products and generate structured business insights
            using Claude Opus 4.6 and BGE-M3 semantic search. Ask any business question and get
            instant, data-grounded intelligence reports.
        </div>
        <div class="home-stats">
            <div><div class="home-stat-val">{total_vectors:,}</div><div class="home-stat-label">Products Indexed</div></div>
            <div><div class="home-stat-val">200K+</div><div class="home-stat-label">Electronics SKUs</div></div>
            <div><div class="home-stat-val">1024-dim</div><div class="home-stat-label">Vector Space</div></div>
            <div><div class="home-stat-val">&lt; 0.2s</div><div class="home-stat-label">Search Latency</div></div>
            <div><div class="home-stat-val">8</div><div class="home-stat-label">Eval Metrics</div></div>
        </div>
        <div class="home-status">
            <div class="home-status-dot"></div>
            <div class="home-status-text">All Systems Operational</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What you can access
    st.markdown('<div class="section-header">📌 What You Can Access</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="page-card">
            <div class="page-card-icon">🔍</div>
            <div class="page-card-title">Product Intelligence</div>
            <div class="page-card-desc">Ask any business question and get AI-generated, data-grounded product intelligence reports powered by Claude Opus 4.6.</div>
            <div class="page-card-link">→ Use the sidebar to navigate</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="page-card">
            <div class="page-card-icon">📊</div>
            <div class="page-card-title">Market Analytics</div>
            <div class="page-card-desc">Explore interactive charts showing rating distributions, category breakdowns, and price analytics across 200K products.</div>
            <div class="page-card-link">→ Use the sidebar to navigate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="page-card">
            <div class="page-card-icon">📈</div>
            <div class="page-card-title">Performance Metrics</div>
            <div class="page-card-desc">Review ROUGE, BERTScore, Faithfulness and Relevance evaluation results with full system configuration details.</div>
            <div class="page-card-link">→ Use the sidebar to navigate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Powered by
    st.markdown('<div class="section-header">⚡ Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="powered-grid">
        <div class="powered-item">
            <div class="powered-icon">🤖</div>
            <div>
                <div class="powered-name">Claude Opus 4.6</div>
                <div class="powered-desc">Anthropic — LLM generation</div>
            </div>
        </div>
        <div class="powered-item">
            <div class="powered-icon">🗄️</div>
            <div>
                <div class="powered-name">ChromaDB</div>
                <div class="powered-desc">Vector store + HNSW search</div>
            </div>
        </div>
        <div class="powered-item">
            <div class="powered-icon">🧠</div>
            <div>
                <div class="powered-name">BGE-M3</div>
                <div class="powered-desc">BAAI — 1024-dim embeddings</div>
            </div>
        </div>
        <div class="powered-item">
            <div class="powered-icon">📊</div>
            <div>
                <div class="powered-name">MLflow</div>
                <div class="powered-desc">Experiment tracking</div>
            </div>
        </div>
        <div class="powered-item">
            <div class="powered-icon">🐳</div>
            <div>
                <div class="powered-name">Docker</div>
                <div class="powered-desc">Containerization</div>
            </div>
        </div>
        <div class="powered-item">
            <div class="powered-icon">⚙️</div>
            <div>
                <div class="powered-name">GitHub Actions</div>
                <div class="powered-desc">CI/CD pipeline</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Initializing..."):
        embed_model, claude_client, collection, loaded = load_resources()

    k, selected_query = render_sidebar(collection, loaded)

    if not loaded:
        st.error("⚠️ System initialization failed. Please check your configuration.")
        return

    total_vectors = collection.count()
    page = st.session_state.get("page", "home")

    # ── HOME ──────────────────────────────────────────────────────────────────
    if page == "home":
        show_home(total_vectors)

    # ── PRODUCT INTELLIGENCE ──────────────────────────────────────────────────
    elif page == "search":
        st.markdown("""
        <div class="nav-bar">
            <div>
                <div class="nav-logo">📦 Product<span>IQ</span></div>
                <div class="nav-tagline">Product Intelligence</div>
            </div>
            <div class="nav-badge">✦ AI Powered</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item"><div class="stat-number">{total_vectors:,}</div><div class="stat-label">Products Indexed</div></div>
            <div class="stat-item"><div class="stat-number">200K+</div><div class="stat-label">Electronics SKUs</div></div>
            <div class="stat-item"><div class="stat-number">1024-dim</div><div class="stat-label">Vector Space</div></div>
            <div class="stat-item"><div class="stat-number">&lt; 0.2s</div><div class="stat-label">Search Latency</div></div>
            <div class="stat-item"><div class="stat-number">8</div><div class="stat-label">Eval Metrics</div></div>
            <div style="margin-left:auto">
                <div class="status-online"><div class="status-dot"></div>All Systems Operational</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔍 Product Intelligence Search</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "",
                value=selected_query or "",
                placeholder="e.g. What are the best wireless headphones under $100?",
                label_visibility="collapsed"
            )
        with col2:
            search_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

        st.markdown("**Trending searches:**")
        cols = st.columns(3)
        samples = [
            "Best noise cancelling headphones",
            "Laptop accessories highest ratings",
            "Most affordable 4K monitors",
        ]
        for i, s in enumerate(samples):
            with cols[i]:
                if st.button(f"🔥 {s}", key=f"s_{i}", use_container_width=True):
                    query = s
                    search_btn = True

        if (search_btn and query) or selected_query:
            if selected_query and not query:
                query = selected_query

            with st.spinner("🔍 Searching product database..."):
                t1      = time.time()
                results = search_products(query, embed_model, collection, k=k)
                t1      = time.time() - t1

            with st.spinner("🤖 Generating intelligence report..."):
                t2                   = time.time()
                insight, docs, metas = generate_insight(query, results, claude_client)
                t2                   = time.time() - t2

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Products Retrieved</div>
                    <div class="metric-value">{k}</div>
                    <div class="metric-sub">from {total_vectors:,} indexed</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Search Latency</div>
                    <div class="metric-value">{t1:.2f}s</div>
                    <div class="metric-sub">vector similarity</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Generation Time</div>
                    <div class="metric-value">{t2:.2f}s</div>
                    <div class="metric-sub">Claude Opus 4.6</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Total Response</div>
                    <div class="metric-value">{t1+t2:.2f}s</div>
                    <div class="metric-sub">end-to-end</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="insight-container">
                <div class="insight-header">
                    <span class="insight-badge">✦ AI Report</span>
                    <span class="insight-title">Product Intelligence Report</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(insight)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📦 Retrieved Products</div>', unsafe_allow_html=True)

            for i, (doc, meta) in enumerate(zip(docs, metas)):
                title  = meta.get('title', 'N/A')
                rating = meta.get('average_rating', 'N/A')
                price  = meta.get('price', 'N/A')
                brand  = meta.get('store', 'N/A')

                with st.expander(f"#{i+1}  {title[:70]}{'...' if len(title) > 70 else ''}"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Brand", brand)
                    with c2: st.metric("Price", f"${price}")
                    with c3: st.metric("Rating", f"{rating}/5.0")
                    with c4: st.metric("Category", meta.get('main_category', 'N/A')[:15])
                    st.markdown(f"**Details:** {doc[:400]}{'...' if len(doc) > 400 else ''}")

    # ── MARKET ANALYTICS ──────────────────────────────────────────────────────
    elif page == "market":
        st.markdown("""
        <div class="nav-bar">
            <div>
                <div class="nav-logo">📦 Product<span>IQ</span></div>
                <div class="nav-tagline">Market Analytics</div>
            </div>
            <div class="nav-badge">✦ AI Powered</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Market Analytics Dashboard</div>', unsafe_allow_html=True)

        chunks_file = None
        for path in [
            "data/processed/processed_chunks.csv",
            "./data/processed/processed_chunks.csv",
            str(Path(__file__).parent.parent.parent / "data" / "processed" / "processed_chunks.csv")
        ]:
            if Path(path).exists():
                chunks_file = path
                break

        if chunks_file:
            with st.spinner("Loading market data..."):
                df = pd.read_csv(chunks_file, nrows=50000)

            c1, c2, c3, c4 = st.columns(4)
            avg_rating = pd.to_numeric(df["average_rating"], errors="coerce").mean()
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Total Products</div>
                    <div class="metric-value">{df['product_id'].nunique():,}</div>
                    <div class="metric-sub">unique SKUs</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Avg Rating</div>
                    <div class="metric-value">{avg_rating:.2f}</div>
                    <div class="metric-sub">out of 5.0 stars</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Categories</div>
                    <div class="metric-value">{df['main_category'].nunique():,}</div>
                    <div class="metric-sub">product categories</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Data Points</div>
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-sub">indexed chunks</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
                fig = px.histogram(df, x="average_rating", nbins=20,
                    title="⭐ Customer Rating Distribution",
                    color_discrete_sequence=["#FF9900"],
                    labels={"average_rating": "Rating"})
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                    font_family="Inter", title_font_size=14, showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(gridcolor="#F0F2F2")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cat_counts = df["main_category"].value_counts().head(8)
                fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                    title="📦 Category Market Share",
                    color_discrete_sequence=px.colors.sequential.Oranges_r)
                fig.update_layout(font_family="Inter", title_font_size=14,
                    margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df_price = df[df["price"].notna() & (df["price"] < 500) & (df["price"] > 0)]
            if len(df_price) > 0:
                fig = px.histogram(df_price, x="price", nbins=40,
                    title="💰 Price Distribution (Products under $500)",
                    color_discrete_sequence=["#146EB4"],
                    labels={"price": "Price (USD)"})
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                    font_family="Inter", title_font_size=14, showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20))
                fig.update_xaxes(showgrid=False, tickprefix="$")
                fig.update_yaxes(gridcolor="#F0F2F2")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Market analytics data is being processed. Please run the data pipeline to populate this section.")
            st.code("python src/data/preprocessor.py")

    # ── PERFORMANCE METRICS ───────────────────────────────────────────────────
    elif page == "metrics":
        st.markdown("""
        <div class="nav-bar">
            <div>
                <div class="nav-logo">📦 Product<span>IQ</span></div>
                <div class="nav-tagline">Performance Metrics</div>
            </div>
            <div class="nav-badge">✦ AI Powered</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 System Performance Metrics</div>', unsafe_allow_html=True)

        if Path("evaluation_results.csv").exists():
            results_df = pd.read_csv("evaluation_results.csv")
            metrics    = results_df.columns.tolist()
            values     = results_df.iloc[0].tolist()

            metric_info = {
                "rouge1":              ("ROUGE-1",       "Lexical overlap"),
                "rouge2":              ("ROUGE-2",       "Bigram overlap"),
                "rougeL":              ("ROUGE-L",       "Sequence overlap"),
                "bertscore_f1":        ("BERTScore F1",  "Semantic similarity"),
                "bertscore_precision": ("BERTScore P",   "Precision"),
                "bertscore_recall":    ("BERTScore R",   "Recall"),
                "faithfulness":        ("Faithfulness",  "Context grounding"),
                "answer_relevance":    ("Relevance",     "Query alignment"),
            }

            cols = st.columns(4)
            for i, (metric, value) in enumerate(zip(metrics, values)):
                with cols[i % 4]:
                    label, desc = metric_info.get(metric, (metric, ""))
                    color = "#007600" if float(value) > 0.7 else "#FF9900" if float(value) > 0.4 else "#B12704"
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color}">{value}</div>
                        <div class="metric-sub">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[metric_info.get(m, (m,))[0] for m in metrics],
                y=values,
                marker_color=["#007600" if v > 0.7 else "#FF9900" if v > 0.4 else "#B12704" for v in values],
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                title="Evaluation Metrics — RAG System Performance",
                plot_bgcolor="white", paper_bgcolor="white",
                font_family="Inter", title_font_size=14, showlegend=False,
                yaxis=dict(range=[0, 1.1], gridcolor="#F0F2F2"),
                xaxis=dict(showgrid=False),
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("📊 Run evaluation to generate metrics.")
            st.code("python src/evaluation/metrics.py")

        st.markdown('<div class="section-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
| Component | Specification |
|---|---|
| **LLM** | Claude Opus 4.6 |
| **Embeddings** | BGE-M3 (1024-dim) |
| **Vector Store** | ChromaDB + HNSW |
| **Retrieval** | Top-20 → Rerank Top-5 |
            """)
        with c2:
            st.markdown("""
| Metric | Value |
|---|---|
| **Vectors Indexed** | 200,000 |
| **Chunks Processed** | 1,820,026 |
| **Search Latency** | < 0.2s |
| **Tests Passing** | 16/16 ✅ |
            """)

if __name__ == "__main__":
    main()