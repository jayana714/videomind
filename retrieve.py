import chromadb
from sentence_transformers import SentenceTransformer

def search(query, db_path="./chroma_db", n_results=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="video_chunks")

    # Turn the question into an embedding, same way we embedded the chunks
    query_embedding = model.encode([query]).tolist()

    # Search Chroma for the most similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    print(f"Query: {query}\n")
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        print(f"Match {i+1} [{meta['start']:.2f}s -> {meta['end']:.2f}s]")
        print(text.strip())
        print("-" * 40)

    return results

if __name__ == "__main__":
    query = input("Ask a question about the video: ")
    search(query)