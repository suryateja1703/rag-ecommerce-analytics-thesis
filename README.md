# RAG-Powered Customer Insight Generation for E-Commerce Using LLMs, Vector Search, and an End-to-End MLOps Pipeline

An intelligent product analytics platform built on the Amazon Electronics dataset, combining semantic vector search, Retrieval-Augmented Generation, and AI-powered natural language querying to make product intelligence more accessible and actionable.

---

## 🌐 Live Demo

| | Link |
|---|---|
| 🚀 **Live Dashboard** | | 🚀 **Live Dashboard** | Run locally — see Quick Start below |
| 💻 **GitHub Repo** | [suryateja1703/rag-ecommerce-analytics-thesis](https://github.com/suryateja1703/rag-ecommerce-analytics-thesis) |
| 📦 **Dataset** | [Amazon Reviews 2023 — UCSD](https://amazon-reviews-2023.github.io) |

---

## 🎯 Overview

**ProductIQ** is an AI-powered product analytics platform that uses Retrieval-Augmented Generation (RAG) with Claude Opus 4.6 to generate structured, executive-ready business insights from 200,000 Amazon Electronics products.

The system uses BGE-M3 sentence embeddings and ChromaDB vector search to retrieve the most semantically relevant products for any business query, then conditions Claude Opus 4.6 to synthesize those products into coherent analytical reports complete with market analysis and strategic recommendations.

Built with a production-first mindset — every component is containerized, tested, tracked, and deployed through an automated CI/CD pipeline.

---

## 🔥 Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Claude Opus 4.6 (Anthropic) |
| **Embeddings** | BGE-M3 (BAAI) — 1024-dim vectors |
| **Vector Store** | ChromaDB + HNSW indexing |
| **RAG Framework** | LangChain |
| **Experiment Tracking** | MLflow |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Dashboard** | Streamlit + Plotly |
| **Evaluation** | ROUGE + BERTScore |

---

## 🏗️ Architecture
Amazon Electronics Metadata (200K products)
↓
Preprocessing Pipeline — 1,820,026 chunks
↓
BGE-M3 Embeddings — 1024-dim vectors
↓
ChromaDB Vector Store — 200K indexed vectors
↓
RAG Pipeline — Retrieve (20) → Rerank (5) → Prompt
↓
Claude Opus 4.6 — Insight Generation
↓
ProductIQ Streamlit Dashboard
↓
MLflow + Docker + GitHub Actions (MLOps)

---

## 📊 Dataset

- **Source:** [Amazon Reviews 2023 — UCSD](https://amazon-reviews-2023.github.io)
- **Category:** Electronics
- **Products:** 200,000 product records
- **Chunks:** 1,820,026 text chunks after preprocessing
- **Vectors:** 200,000 indexed in ChromaDB
- **Fields:** Title, Brand, Category, Price, Rating, Description, Features, Specifications

---

## 📈 Evaluation Results

| Metric | Score | Description |
|---|---|---|
| **ROUGE-1** | 0.4121 | Unigram lexical overlap |
| **ROUGE-2** | 0.2051 | Bigram lexical overlap |
| **ROUGE-L** | 0.4121 | Longest common subsequence |
| **BERTScore Precision** | 0.8918 | Semantic precision |
| **BERTScore Recall** | 0.9361 | Semantic recall |
| **BERTScore F1** | 0.9131 | Semantic similarity |
| **Faithfulness** | 0.5567 | Context grounding |
| **Answer Relevance** | 0.2857 | Query alignment |

---

## ✅ System Stats

| Metric | Value |
|---|---|
| **Vectors Indexed** | 200,000 |
| **Chunks Processed** | 1,820,026 |
| **Tests Passing** | 16 / 16 ✅ |
| **Search Latency** | < 0.2s |
| **Test Suite Time** | 11.18s |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/suryateja1703/rag-ecommerce-analytics-thesis
cd rag-ecommerce-analytics-thesis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
PRIMARY_MODEL=claude-opus-4-6
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_PERSIST_PATH=./embeddings
DATA_PATH=./data/processed
MLFLOW_TRACKING_URI=./mlruns
RETRIEVAL_K=5
TEMPERATURE=0.3
```

### 3. Run Pipeline
```bash
# Step 1 — Preprocess data
python src/data/preprocessor.py

# Step 2 — Embed chunks into ChromaDB
python src/rag/embedder.py

# Step 3 — Test RAG pipeline
python src/rag/pipeline.py

# Step 4 — Run evaluation
python src/evaluation/metrics.py

# Step 5 — Launch dashboard
streamlit run src/app/dashboard.py
```

### 4. Run with Docker
```bash
docker-compose up --build
```

### 5. Run Tests
```bash
pytest tests/ -v
```

---

## 📁 Project Structure
rag-ecommerce-analytics-thesis/
├── src/
│   ├── data/
│   │   └── preprocessor.py        # Data ingestion & chunking pipeline
│   ├── rag/
│   │   ├── embedder.py            # BGE-M3 embedding + ChromaDB indexing
│   │   └── pipeline.py            # RAG chain — retrieve, rerank, generate
│   ├── evaluation/
│   │   └── metrics.py             # ROUGE + BERTScore evaluation
│   └── app/
│       └── dashboard.py           # ProductIQ Streamlit dashboard
├── tests/
│   ├── test_preprocessor.py       # Preprocessing unit tests
│   └── test_pipeline.py           # Pipeline unit tests
├── embeddings/                    # ChromaDB persistent vector store
├── data/
│   ├── raw/                       # Raw Amazon metadata
│   └── processed/                 # Preprocessed chunks CSV
├── mlruns/                        # MLflow experiment artifacts
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── Dockerfile                     # Container build spec
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
└── README.md

---

## 👨‍💻 Author

**Sai Surya Teja Medisetty** — MS in Data Science, UMass Dartmouth

GitHub: [@suryateja1703](https://github.com/suryateja1703)

---

## 📄 License

This project is for academic purposes — MS Data Science Thesis, UMass Dartmouth 2026.
