import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="video_keyframes")

print(f"Collection 'video_keyframes' currently has {collection.count()} items.\n")

results = collection.get(limit=20)
for i, (doc_id, doc) in enumerate(zip(results["ids"], results["documents"])):
    print(f"[{doc_id}] {doc[:150]}...")
    print("-" * 40)