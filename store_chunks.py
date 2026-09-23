import chromadb
from sentence_transformers import SentenceTransformer
import json

def store_chunks(chunks_path="chunks.json", db_path="./chroma_db"):
    # Load chunks from Step 3
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Load the same embedding model from Step 4
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up a local, persistent Chroma database
    client = chromadb.PersistentClient(path=db_path)
    # Start fresh each time -- without this, processing a second video would crash
    # trying to add chunk IDs (chunk_0, chunk_1...) that already exist from the first one.
    try:
        client.delete_collection(name="video_chunks")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="video_chunks")

    # Prepare data for Chroma
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"start": c["start"], "end": c["end"]} for c in chunks]

    # Add everything to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in Chroma at {db_path}")
    print(f"Collection now has {collection.count()} items")

if __name__ == "__main__":
    store_chunks()