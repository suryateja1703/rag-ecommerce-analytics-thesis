import os
import json
import gzip
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

DATA_PATH     = os.getenv("DATA_PATH", "./data/processed")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
RAW_DATA_PATH = "./data/raw"
MAX_RECORDS   = 200_000

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s.,!?'$%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_rich_text(record):
    """Build rich text from metadata fields for RAG."""
    parts = []

    title = record.get("title", "")
    if title:
        parts.append(f"Product: {clean_text(title)}")

    store = record.get("store", "")
    if store:
        parts.append(f"Brand/Store: {clean_text(store)}")

    categories = record.get("categories", [])
    if categories:
        parts.append(f"Category: {' > '.join(categories)}")

    price = record.get("price")
    if price:
        parts.append(f"Price: ${price}")

    avg_rating    = record.get("average_rating")
    rating_number = record.get("rating_number")
    if avg_rating:
        parts.append(f"Average Rating: {avg_rating}/5.0 ({rating_number} ratings)")

    description = record.get("description", [])
    if description:
        desc_text = " ".join(description) if isinstance(description, list) else description
        desc_text = clean_text(desc_text)
        if desc_text:
            parts.append(f"Description: {desc_text}")

    features = record.get("features", [])
    if features:
        features_text = " | ".join([clean_text(f) for f in features if f])
        if features_text:
            parts.append(f"Features: {features_text}")

    details = record.get("details", {})
    if details and isinstance(details, dict):
        detail_parts = []
        for k, v in details.items():
            if isinstance(v, dict):
                continue
            detail_parts.append(f"{k}: {v}")
        if detail_parts:
            parts.append(f"Specifications: {' | '.join(detail_parts[:8])}")

    return "\n".join(parts)

def load_metadata(filepath, max_records=MAX_RECORDS):
    """Load metadata from jsonl or jsonl.gz file."""
    print(f"\n📂 Loading metadata from {Path(filepath).name}...")
    records = []

    opener = gzip.open if filepath.endswith(".gz") else open
    mode   = "rt" if filepath.endswith(".gz") else "r"

    with opener(filepath, mode, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading records")):
            if i >= max_records:
                break
            try:
                record = json.loads(line)
                records.append(record)
            except:
                continue

    print(f"✅ Loaded {len(records):,} product records")
    return records

def process_metadata(records):
    """Convert metadata records into rich text documents."""
    print(f"\n🔨 Building rich text from {len(records):,} records...")
    processed = []

    for record in tqdm(records, desc="Processing"):
        rich_text = build_rich_text(record)
        if len(rich_text.split()) < 10:
            continue
        processed.append({
            "product_id"    : str(record.get("parent_asin", "")),
            "title"         : clean_text(record.get("title", "")),
            "store"         : clean_text(record.get("store", "")),
            "main_category" : record.get("main_category", ""),
            "categories"    : " > ".join(record.get("categories", [])),
            "price"         : record.get("price"),
            "average_rating": record.get("average_rating"),
            "rating_number" : record.get("rating_number"),
            "full_text"     : rich_text,
        })

    print(f"✅ {len(processed):,} valid product records processed")
    return pd.DataFrame(processed)

def chunk_documents(df):
    """Split product metadata into chunks for RAG."""
    print(f"\n✂️  Chunking {len(df):,} products (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n", ". ", " ", ""]
    )
    chunks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text_chunks = splitter.split_text(row["full_text"])
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "chunk_id"      : f"{row['product_id']}_{i}",
                "product_id"    : row["product_id"],
                "chunk_text"    : chunk,
                "chunk_index"   : i,
                "title"         : row["title"],
                "store"         : row["store"],
                "main_category" : row["main_category"],
                "categories"    : row["categories"],
                "price"         : str(row["price"]) if row["price"] else "N/A",
                "average_rating": str(row["average_rating"]) if row["average_rating"] else "N/A",
                "rating_number" : str(row["rating_number"]) if row["rating_number"] else "N/A",
            })
    print(f"✅ Created {len(chunks):,} chunks!")
    return chunks

def save_chunks(chunks, output_path):
    """Save processed chunks to CSV."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df_chunks   = pd.DataFrame(chunks)
    output_file = os.path.join(output_path, "processed_chunks.csv")
    df_chunks.to_csv(output_file, index=False)
    print(f"\n💾 Saved {len(chunks):,} chunks to {output_file}")
    return output_file

def run_pipeline():
    print("=" * 60)
    print("🚀 Metadata RAG Pipeline — Claude Opus 4.6 Thesis")
    print(f"📊 Scale: Up to {MAX_RECORDS:,} products per file")
    print("=" * 60)

    # Get unique files only
    seen = set()
    raw_files = []
    for f in (list(Path(RAW_DATA_PATH).glob("meta_*.jsonl")) +
              list(Path(RAW_DATA_PATH).glob("meta_*.jsonl.gz"))):
        if f.name not in seen:
            seen.add(f.name)
            raw_files.append(f)

    if not raw_files:
        print("❌ No metadata files found in data/raw/")
        print("👉 Make sure files start with 'meta_'")
        return

    print(f"📁 Found: {[f.name for f in raw_files]}")

    all_chunks = []
    for filepath in raw_files:
        print(f"\n📄 Processing: {filepath.name}")
        records = load_metadata(str(filepath))
        df      = process_metadata(records)
        chunks  = chunk_documents(df)
        all_chunks.extend(chunks)

    output = save_chunks(all_chunks, DATA_PATH)

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"📊 Total chunks created : {len(all_chunks):,}")
    print(f"📁 Output saved to      : {output}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()