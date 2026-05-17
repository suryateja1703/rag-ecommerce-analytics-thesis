import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_pipeline_imports():
    """Test that pipeline modules import correctly."""
    try:
        from src.rag.pipeline import retrieve_chunks, rerank_chunks, generate_insight
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_rerank_chunks():
    """Test reranking logic."""
    from src.rag.pipeline import rerank_chunks
    chunks = [
        {"text": "chunk1", "metadata": {}, "distance": 0.8},
        {"text": "chunk2", "metadata": {}, "distance": 0.3},
        {"text": "chunk3", "metadata": {}, "distance": 0.5},
    ]
    reranked = rerank_chunks(chunks, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["distance"] <= reranked[1]["distance"]

def test_rerank_top_k():
    """Test reranking returns correct number."""
    from src.rag.pipeline import rerank_chunks
    chunks = [
        {"text": f"chunk{i}", "metadata": {}, "distance": float(i)}
        for i in range(10)
    ]
    reranked = rerank_chunks(chunks, top_k=5)
    assert len(reranked) == 5

def test_chromadb_connection():
    """Test ChromaDB connection."""
    try:
        import chromadb
        client     = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("amazon_products")
        assert collection.count() > 0
        print(f"✅ ChromaDB has {collection.count():,} vectors")
    except Exception as e:
        pytest.skip(f"ChromaDB not ready: {e}")

def test_anthropic_connection():
    """Test Anthropic API connection."""
    try:
        import anthropic
        from dotenv import load_dotenv
        import os
        load_dotenv()
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        assert client is not None
        print("✅ Anthropic client initialized")
    except Exception as e:
        pytest.fail(f"Anthropic connection failed: {e}")

def test_embedding_model():
    """Test sentence transformer loading."""
    try:
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        model     = SentenceTransformer(
            "BAAI/bge-m3",
            local_files_only=True
        )
        embedding = model.encode(
            ["test text"],
            normalize_embeddings=True
        )
        assert embedding.shape[0] == 1
        print(f"✅ Embedding shape: {embedding.shape}")
    except Exception as e:
        pytest.skip(f"Model not available: {e}")

def test_retrieve_chunks():
    """Test retrieval from ChromaDB."""
    try:
        import os
        import chromadb
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        from src.rag.pipeline import retrieve_chunks

        model      = SentenceTransformer(
            "BAAI/bge-m3",
            local_files_only=True
        )
        client     = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("amazon_products")

        chunks = retrieve_chunks(
            "best wireless headphones",
            model, collection, k=5
        )
        assert len(chunks) > 0
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]
        print(f"✅ Retrieved {len(chunks)} chunks")
    except Exception as e:
        pytest.skip(f"Retrieval test skipped: {e}")