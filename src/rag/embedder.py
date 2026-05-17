import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Force offline mode — use cached model
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PERSIST_PATH", "./embeddings")
DATA_PATH       = os.getenv("DATA_PATH", "./data/processed")
EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE      = 32
MAX_CHUNKS      = 200000  # Safe limit — finishes in ~15 mins from current position

def load_chunks(filepath):
    print(f"\n📂 Loading chunks from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} total chunks")
    # Limit to MAX_CHUNKS
    df = df.head(MAX_CHUNKS)
    print(f"📊 Using {len(df):,} chunks (limit: {MAX_CHUNKS:,})")
    return df

def get_embedding_model():
    print(f"\n🤖 Loading BGE-M3 from local cache...")
    model = SentenceTransformer(
        "BAAI/bge-m3",
        local_files_only=True  # Use cached model only — no download!
    )
    print("✅ BGE-M3 loaded from cache!")
    return model

def get_chroma_collection():
    print(f"\n🗄️  Initializing ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="amazon_products",
        metadata={"hnsw:space": "cosine"}
    )
    existing = collection.count()
    print(f"✅ ChromaDB ready!")
    print(f"💾 Already saved: {existing:,} vectors")
    return collection

def embed_and_store(df, model, collection):
    # Resume — skip already embedded chunks
    existing_count = collection.count()

    if existing_count >= len(df):
        print(f"\n✅ All {len(df):,} chunks already embedded!")
        print(f"💾 ChromaDB has {existing_count:,} vectors")
        print("🚀 Ready for RAG pipeline!")
        return collection

    if existing_count > 0:
        print(f"\n⚡ Resuming from chunk {existing_count:,}...")
        df = df.iloc[existing_count:]
        print(f"📊 Remaining to embed: {len(df):,} chunks")
        print(f"⏱️  Estimated time: ~{len(df)//3600 + 1} hours")
    else:
        print(f"\n🔢 Starting fresh embedding of {len(df):,} chunks...")

    texts     = df["chunk_text"].fillna("").tolist()
    chunk_ids = df["chunk_id"].astype(str).tolist()

    # Build metadata safely
    meta_cols = ["product_id", "title", "store",
                 "main_category", "categories",
                 "price", "average_rating", "rating_number"]
    available  = [c for c in meta_cols if c in df.columns]
    metadatas  = df[available].fillna("N/A").astype(str).to_dict("records")

    total_embedded = 0
    failed_batches = 0

    with tqdm(total=len(texts), desc="Embedding & storing") as pbar:
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts     = texts[i:i+BATCH_SIZE]
            batch_ids       = chunk_ids[i:i+BATCH_SIZE]
            batch_metadatas = metadatas[i:i+BATCH_SIZE]

            try:
                # Generate embeddings
                embeddings = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).tolist()

                # Store in ChromaDB
                collection.add(
                    documents=batch_texts,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                total_embedded += len(batch_texts)
                pbar.update(len(batch_texts))

                # Report progress every 10K
                if total_embedded % 10000 == 0:
                    total_so_far = existing_count + total_embedded
                    print(f"\n💾 Progress: {total_embedded:,} new | {total_so_far:,} total in ChromaDB")

            except Exception as e:
                failed_batches += 1
                print(f"\n⚠️  Batch failed: {e}")
                pbar.update(len(batch_texts))
                continue

    print(f"\n✅ Done!")
    print(f"📊 New chunks embedded  : {total_embedded:,}")
    print(f"⚠️  Failed batches       : {failed_batches}")
    print(f"💾 Total in ChromaDB    : {collection.count():,}")
    return collection

def run_embedder():
    print("=" * 60)
    print("🚀 Embedding Pipeline — BGE-M3 + ChromaDB")
    print(f"📊 Target: {MAX_CHUNKS:,} chunks | Auto-resume enabled")
    print("=" * 60)

    # Check chunks file
    chunks_file = os.path.join(DATA_PATH, "processed_chunks.csv")
    if not Path(chunks_file).exists():
        print("❌ processed_chunks.csv not found!")
        print("👉 Run preprocessor.py first!")
        return

    df         = load_chunks(chunks_file)
    model      = get_embedding_model()
    collection = get_chroma_collection()
    collection = embed_and_store(df, model, collection)

    print("\n" + "=" * 60)
    print("✅ Embedding Pipeline Complete!")
    print(f"📊 Total vectors in ChromaDB: {collection.count():,}")
    print("🚀 Ready for RAG pipeline!")
    print("=" * 60)

if __name__ == "__main__":
    run_embedder()