import os
from anthropic import Anthropic
from dotenv import load_dotenv
from retrieve import search

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are answering questions about a video using ONLY the transcript excerpts given to you as context.
- Base your answer strictly on the provided excerpts. If the excerpts don't contain the answer, say so instead of guessing.
- Reference approximate timestamps (MM:SS) so the user can jump to that part of the video.
- Be concise and direct — this is replacing watching the video, not summarizing forever."""


def format_timestamp(seconds):
    """Convert raw seconds (e.g. 132.32) into MM:SS for a human-readable citation."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_context(results):
    """Turn Chroma's retrieval results into a numbered, timestamped context block for the prompt."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    blocks = []
    for i, (text, meta) in enumerate(zip(documents, metadatas)):
        start = format_timestamp(meta["start"])
        end = format_timestamp(meta["end"])
        blocks.append(f"[Excerpt {i + 1}, {start}-{end}]\n{text.strip()}")

    return "\n\n".join(blocks)


def answer_question(query, n_results=3, model="claude-haiku-4-5-20251001"):
    # Step 6: retrieve the most relevant chunks (already built)
    results = search(query, n_results=n_results)
    context = format_context(results)

    # Step 7: generate a real answer grounded in those chunks
    user_message = f"Video transcript excerpts:\n\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    print(f"\nAnswer:\n{answer}\n")
    return answer


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your ANTHROPIC_API_KEY environment variable first, e.g.:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
    else:
        query = input("Ask a question about the video: ")
        answer_question(query)