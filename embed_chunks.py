from sentence_transformers import SentenceTransformer
import json

def embed_chunks(chunks_path="chunks.json", model_name="all-MiniLM-L6-v2"):
    # Load the chunks we made in Step 3
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Load a small, free, local embedding model
    model = SentenceTransformer(model_name)

    # Turn each chunk's text into an embedding (a list of numbers)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    print(f"Created {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} numbers")
    print("\nFirst 5 numbers of chunk 0's embedding:")
    print(embeddings[0][:5])

    return chunks, embeddings

if __name__ == "__main__":
    embed_chunks()