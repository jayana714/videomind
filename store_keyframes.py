import chromadb
from sentence_transformers import SentenceTransformer
import json

def store_keyframes(keyframes_path="keyframes/keyframes.json", db_path="./chroma_db"):
    """Stores keyframe descriptions in their own Chroma collection, separate from
    video_chunks (the audio transcript). Keeping audio and visual data in separate
    collections -- rather than mixing them into one -- is what makes Step 4's
    LangGraph routing possible: the router can query one, the other, or both,
    per question, instead of always searching everything at once."""
    with open(keyframes_path) as f:
        keyframes = json.load(f)

    # Same embedding model as the audio chunks (store_chunks.py) -- doesn't need to
    # match, but keeping it consistent means one less moving piece to reason about.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)
    # Same reasoning as store_chunks.py -- start fresh so re-processing a video
    # doesn't collide with a previous video's keyframe IDs.
    try:
        client.delete_collection(name="video_keyframes")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="video_keyframes")

    descriptions = [kf["description"] for kf in keyframes]
    embeddings = model.encode(descriptions).tolist()
    ids = [f"keyframe_{i}" for i in range(len(keyframes))]
    metadatas = [{"timestamp": kf["timestamp"], "path": kf["path"]} for kf in keyframes]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=descriptions,
        metadatas=metadatas,
    )

    print(f"Stored {len(keyframes)} keyframe descriptions in Chroma at {db_path}")
    print(f"Collection now has {collection.count()} items")

if __name__ == "__main__":
    store_keyframes()