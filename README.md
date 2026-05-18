
# RAG-Powered Customer Insight Generation for E-Commerce
## Using LLMs, Vector Search, and an End-to-End MLOps Pipeline

**University of Massachusetts Dartmouth**
**Master of Science in Data Science | 2026**
**Author: Surya [Last Name]**

---

## 🎯 Overview

An intelligent analytics platform that leverages Retrieval-Augmented 
Generation (RAG) with Claude Opus 4.6 to generate business insights 
from Amazon Electronics product metadata at enterprise scale.

The system retrieves semantically relevant product information using 
BGE-M3 embeddings and ChromaDB vector search, then generates 
structured analytical insights using Claude Opus 4.6.

---

## 🏗️ Architecture
Amazon Metadata (200K products)
↓
Preprocessing Pipeline (1.82M chunks)
↓
BGE-M3 Embeddings + ChromaDB (200K vectors)
↓
LangChain RAG Chain
↓
Claude Opus 4.6 (Insight Generation)
↓
MLflow + Docker + GitHub Actions (MLOps)
↓
Streamlit Dashboard (Live Demo)
---

## 🔥 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Claude Opus 4.6 (Anthropic) |
| **Embeddings** | BGE-M3 (BAAI) |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **Experiment Tracking** | MLflow |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Dashboard** | Streamlit + Plotly |
| **Evaluation** | ROUGE + BERTScore + DeepEval + RAGAS |

---

## 📊 Dataset

- **Source:** Amazon Electronics Metadata (UCSD 2023)
- **Products:** 200,000 electronics products
- **Chunks:** 1,820,026 text chunks
- **Vectors:** 200,000 indexed in ChromaDB
- **Fields:** Title, Brand, Category, Price, Rating, Description, Features

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/rag-ecommerce-analytics-thesis
cd rag-ecommerce-analytics-thesis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Create .env file
ANTHROPIC_API_KEY=your_key_here
PRIMARY_MODEL=claude-opus-4-6
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_PERSIST_PATH=./embeddings
DATA_PATH=./data/processed
MLFLOW_TRACKING_URI=./mlruns
```

### 3. Run Pipeline
```bash
# Step 1 — Preprocess data
python src/data/preprocessor.py

# Step 2 — Embed chunks
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

## 📈 Evaluation Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | TBD |
| ROUGE-2 | TBD |
| ROUGE-L | TBD |
| BERTScore F1 | TBD |
| Faithfulness | TBD |
| Answer Relevance | TBD |

---

## 📁 Project Structure
thesis/
├── src/
│   ├── data/
│   │   └── preprocessor.py     # Data pipeline
│   ├── rag/
│   │   ├── embedder.py         # BGE-M3 + ChromaDB
│   │   └── pipeline.py         # RAG chain
│   ├── evaluation/
│   │   └── metrics.py          # ROUGE + BERTScore
│   └── app/
│       └── dashboard.py        # Streamlit UI
├── tests/
│   ├── test_preprocessor.py
│   └── test_pipeline.py
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed chunks
├── embeddings/                 # ChromaDB vectors
├── mlruns/                     # MLflow experiments
├── .github/workflows/
│   └── ci-cd.yml              # GitHub Actions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
---

## 🎓 Academic Information

- **Institution:** University of Massachusetts Dartmouth
- **Program:** Master of Science in Data Science
- **Department:** Computer and Information Science
- **Thesis Advisor:** 
- **Year:** 2026
=======
# rag-ecommerce-analytics-thesis
RAG-Powered Customer Insight Generation for E-Commerce — UMass Dartmouth MS Thesis 2026
>>>>>>> 74d4fb31914c0be262d7e2788c7f7cd2ce1393dc
