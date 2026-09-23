import json

def chunk_segments(segments, max_chunk_duration=40):
    """Groups small Whisper segments into bigger chunks (~max_chunk_duration seconds each),
    keeping track of the start/end time of each chunk."""
    chunks = []
    current_chunk = {"start": None, "end": None, "text": ""}

    for seg in segments:
        if current_chunk["start"] is None:
            current_chunk["start"] = seg["start"]

        current_chunk["text"] += seg["text"]
        current_chunk["end"] = seg["end"]

        # If this chunk has grown long enough, close it and start a new one
        if current_chunk["end"] - current_chunk["start"] >= max_chunk_duration:
            chunks.append(current_chunk)
            current_chunk = {"start": None, "end": None, "text": ""}

    # Add any leftover chunk at the end
    if current_chunk["text"]:
        chunks.append(current_chunk)

    return chunks

if __name__ == "__main__":
    with open("transcript.json") as f:
        segments = json.load(f)

    chunks = chunk_segments(segments)

    print(f"Created {len(chunks)} chunks from {len(segments)} segments\n")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: [{c['start']:.2f}s -> {c['end']:.2f}s]")
        print(c["text"].strip())
        print("-" * 40)

    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"\nSaved {len(chunks)} chunks to chunks.json")