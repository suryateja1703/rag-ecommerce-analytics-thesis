import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocessor import clean_text, build_rich_text

def test_clean_text_basic():
    """Test basic text cleaning."""
    result = clean_text("Hello World!")
    assert isinstance(result, str)
    assert len(result) > 0

def test_clean_text_html():
    """Test HTML removal."""
    result = clean_text("<b>Bold text</b>")
    assert "<b>" not in result
    assert "Bold text" in result

def test_clean_text_url():
    """Test URL removal."""
    result = clean_text("Visit https://amazon.com for deals")
    assert "https://" not in result

def test_clean_text_empty():
    """Test empty string handling."""
    result = clean_text("")
    assert result == ""

def test_clean_text_none():
    """Test None handling."""
    result = clean_text(None)
    assert result == ""

def test_build_rich_text_basic():
    """Test rich text building from metadata."""
    record = {
        "title"         : "Sony Headphones",
        "store"         : "Sony",
        "categories"    : ["Electronics", "Headphones"],
        "price"         : 99.99,
        "average_rating": 4.5,
        "rating_number" : 1000,
        "description"   : ["Great sound quality"],
        "features"      : ["Noise cancellation", "30hr battery"],
        "details"       : {"Color": "Black"},
        "parent_asin"   : "B001234"
    }
    result = build_rich_text(record)
    assert "Sony Headphones" in result
    assert "Sony" in result
    assert "99.99" in result
    assert "4.5" in result

def test_build_rich_text_missing_fields():
    """Test rich text with missing fields."""
    record = {"title": "Test Product", "parent_asin": "B000"}
    result = build_rich_text(record)
    assert "Test Product" in result
    assert isinstance(result, str)

def test_processed_chunks_exist():
    """Test that processed chunks file exists."""
    chunks_file = Path("data/processed/processed_chunks.csv")
    assert chunks_file.exists(), "Run preprocessor.py first!"

def test_processed_chunks_structure():
    """Test structure of processed chunks."""
    chunks_file = Path("data/processed/processed_chunks.csv")
    if chunks_file.exists():
        df = pd.read_csv(chunks_file, nrows=100)
        assert "chunk_id" in df.columns
        assert "chunk_text" in df.columns
        assert "product_id" in df.columns
        assert len(df) > 0