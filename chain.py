import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Same embedding model from Steps 4-6, now wrapped as a LangChain component.
# Using the identical model name keeps it compatible with the existing chroma_db —
# no need to re-embed or re-store anything from Step 5.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_retriever():
    """Connect (or reconnect) to whatever is currently stored in Chroma."""
    vectorstore = Chroma(
        collection_name="video_chunks",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    # k=5 missed a passing GraphQL mention buried in a REST-limitations chunk when
    # asked directly about GraphQL, even though that same chunk surfaced fine for a
    # differently-worded question. Bumped to 8 to improve recall on brief/passing
    # mentions, at the cost of slightly more (and slightly noisier) context per answer.
    return vectorstore.as_retriever(search_kwargs={"k": 8})


retriever = build_retriever()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

# Separate, temperature=0 model specifically for classification tasks (intent,
# routing). Generation tasks (summaries, answers) benefit from some natural
# variation at default temperature; classification tasks need to give the SAME
# answer for the SAME input every time -- without this, identical questions
# ("give me a summary") were observed getting classified differently across
# separate calls, which is a real reliability problem, not a rare edge case.
classifier_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=10, temperature=0)

# Separate model just for full-video summaries. 1024 tokens (the shared `llm`'s cap)
# is plenty for a normal Q&A answer but was silently truncating summaries mid-sentence
# on long videos with lots of ground to cover (transcript + keyframes both folded in) --
# the API just stops generating once it hits the cap, with no error, so it looked like
# a content bug until the raw output showed it cutting off with no closing punctuation.
# Kept separate from `llm` so ordinary Q&A calls stay fast/cheap and this only costs
# more on the summary path where it's actually needed.
summary_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=4096)

SYSTEM_PROMPT = """You are answering questions about a video using ONLY the transcript excerpts given to you as context.
- Base your answer strictly on the provided excerpts. If the excerpts don't contain the answer, say so instead of guessing.
- Reference approximate timestamps (MM:SS) so the user can jump to that part of the video.
- Answer like you're explaining it to a friend, in plain conversational sentences. No headers, no bold text, no bullet lists unless the steps are genuinely sequential and a list is clearer than prose."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Video transcript excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def format_timestamp(seconds):
    """Convert raw seconds (e.g. 132.32) into MM:SS for a human-readable citation."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_docs(docs):
    """Turn LangChain's retrieved Documents into a numbered, timestamped context block."""
    blocks = []
    for i, doc in enumerate(docs):
        start = format_timestamp(doc.metadata["start"])
        end = format_timestamp(doc.metadata["end"])
        blocks.append(f"[Excerpt {i + 1}, {start}-{end}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


# Step 8: the LangChain refactor. Retrieval, prompting, and generation are now one
# composable chain instead of manual function calls — this is what "using LangChain"
# actually means, rather than just calling an LLM API directly.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def refresh():
    """Call this after storing a new video, so the chat connects to the new
    data instead of the stale connection from whatever was loaded at startup."""
    global retriever, chain
    retriever = build_retriever()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    _summary_cache.clear()  # old cached summaries belong to the previous video


_summary_cache = {}


def _video_fingerprint(chunks_path="chunks.json"):
    """A cheap way to tell whether two summary requests are about the same video --
    if chunks.json hasn't changed, it's the same video, so a cached answer is safe to reuse."""
    with open(chunks_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_cached_summary(question):
    key = (_video_fingerprint(), question.strip().lower())
    return _summary_cache.get(key)


def cache_summary(question, answer, verification):
    key = (_video_fingerprint(), question.strip().lower())
    _summary_cache[key] = (answer, verification)


CONDENSE_SYSTEM_PROMPT = """Given a conversation history and a new follow-up question, rewrite the
follow-up question as a standalone question that includes whatever context from the history it
depends on. If the follow-up question is already standalone (doesn't rely on prior messages),
return it completely unchanged. Output ONLY the rewritten question -- no explanation, no quotes."""

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM_PROMPT),
    ("human", "Conversation so far:\n{history}\n\nFollow-up question: {question}\n\nStandalone question:"),
])

condense_chain = condense_prompt | llm | StrOutputParser()


def condense_question(question, history_text):
    """Turns a context-dependent follow-up ('can you say more about that?') into a
    standalone question the retriever can actually search on. Skips the extra LLM
    call entirely when there's no history yet, since there's nothing to condense."""
    if not history_text.strip():
        return question
    return condense_chain.invoke({"history": history_text, "question": question}).strip()


INTENT_SYSTEM_PROMPT = """Classify the user's question about a video as either:
- "summary" -- they want a broad overview, summary, main points, key takeaways, or a
  "what is this video about" style answer covering the whole video
- "specific" -- they want an answer to one narrow, specific detail or fact

Respond with exactly one word: summary or specific."""

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "{question}"),
])

intent_chain = intent_prompt | classifier_llm | StrOutputParser()


def classify_intent(question):
    result = intent_chain.invoke({"question": question}).strip().lower()
    return "summary" if "summary" in result else "specific"


SUMMARY_SYSTEM_PROMPT = """You are summarizing an entire video. You may be given the full spoken
transcript (audio), a set of visual descriptions of what's shown on screen at various timestamps,
or both.
- Cover the whole video, start to finish -- not just the beginning.
- When both spoken and visual content are provided, weave them together naturally at each point
  in the timeline (what was said AND what was on screen at that moment), rather than summarizing
  them as two separate, disconnected sections.
- Include concrete specifics from the visual content where available -- exact layer/file/field
  names, exact button or menu labels, specific values shown in a table -- not just a general
  description of the action being performed. If a name or label is visible on screen, use the
  actual name, not a paraphrase of what it represents.
- Structure your answer as a short overview paragraph, followed by the main points covered, each
  with an approximate timestamp so the person can jump to that part if they want.
- Base everything strictly on the content provided. Do not add outside knowledge about the topic.
- Be genuinely useful, not padded -- someone should be able to decide what to skip and what to
  watch in full after reading this."""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM_PROMPT),
    ("human", "Full video transcript, in order:\n\n{transcript}\n\nRequest: {question}"),
])

summary_chain = summary_prompt | summary_llm | StrOutputParser()


def format_chunks(chunks):
    """Same timestamp formatting as format_docs(), but for raw chunk dicts
    straight from chunks.json rather than LangChain Document objects."""
    blocks = []
    for c in chunks:
        start = format_timestamp(c["start"])
        end = format_timestamp(c["end"])
        blocks.append(f"[{start}-{end}]\n{c['text'].strip()}")
    return "\n\n".join(blocks)


def format_keyframes(keyframes):
    """Same idea as format_chunks(), but for keyframe descriptions instead of
    transcript text -- timestamped blocks ready to drop into a prompt."""
    blocks = []
    for kf in keyframes:
        ts = format_timestamp(kf["timestamp"])
        blocks.append(f"[{ts}, on-screen]\n{kf['description'].strip()}")
    return "\n\n".join(blocks)


def build_full_context(chunks_path="chunks.json", keyframes_path="keyframes/keyframes.json"):
    """Builds the complete context for a summary: audio transcript alone if no
    keyframes exist yet (e.g. a video processed before v2's visual pipeline ran),
    or audio + visual combined when both are available."""
    with open(chunks_path) as f:
        chunks = json.load(f)
    full_transcript = format_chunks(chunks)

    if not os.path.exists(keyframes_path):
        print("[build_full_context: no keyframes.json found — using audio only]")
        return full_transcript

    with open(keyframes_path) as f:
        keyframes = json.load(f)
    visual_context = format_keyframes(keyframes)
    print(f"[build_full_context: including {len(keyframes)} keyframe descriptions alongside audio]")

    return (
        f"SPOKEN CONTENT (audio transcript):\n\n{full_transcript}\n\n"
        f"VISUAL CONTENT (what's shown on screen):\n\n{visual_context}"
    )


def summarize_video(question, chunks_path="chunks.json", keyframes_path="keyframes/keyframes.json"):
    """Used for broad questions (summaries, overviews, key takeaways) where the normal
    top-k retrieval can't give the full picture. Feeds the ENTIRE audio transcript, plus
    visual keyframe descriptions when available, instead of just the most relevant few
    chunks -- fine even for long videos, since this is still far short of the model's
    context limit."""
    full_context = build_full_context(chunks_path, keyframes_path)
    return summary_chain.invoke({"transcript": full_context, "question": question})


SUMMARY_VERIFICATION_SYSTEM_PROMPT = """You are fact-checking a video summary against the video's
full transcript. Check two separate things:
1. Content: is each claim in the summary actually supported by the transcript?
2. Timestamps: for each claim's cited timestamp, does that topic genuinely appear at or near
   that point in the transcript? A summary can restate a topic accurately but attribute it to
   the wrong timestamp (e.g. citing an early preview mention instead of where it's actually
   covered in depth) -- treat that as a real error, not a minor one.

Respond in exactly this format:
Grounded: yes or no
Explanation: one or two sentences. If "no", name the specific claim or timestamp that's wrong."""

summary_verification_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_VERIFICATION_SYSTEM_PROMPT),
    ("human", "Full video transcript, in order:\n\n{transcript}\n\nGenerated summary:\n{summary}"),
])

summary_verification_chain = summary_verification_prompt | llm | StrOutputParser()


def verify_summary(summary, chunks_path="chunks.json", keyframes_path="keyframes/keyframes.json"):
    """Same idea as Step 9's verify.py, but checked against the WHOLE audio+visual
    context (matching what summarize_video actually used) instead of a few retrieved
    chunks, and specifically instructed to catch timestamp-attribution drift."""
    full_context = build_full_context(chunks_path, keyframes_path)
    return summary_verification_chain.invoke({"transcript": full_context, "summary": summary})


CORRECTION_SYSTEM_PROMPT = """You wrote a video summary, and a fact-checker found a specific
problem with it. Fix ONLY the issue described -- don't rewrite unrelated parts, don't shorten
or restructure sections that weren't flagged. Use the full transcript to get the correction
right. Output the complete corrected summary, not just the fixed portion."""

correction_prompt = ChatPromptTemplate.from_messages([
    ("system", CORRECTION_SYSTEM_PROMPT),
    ("human", "Full video transcript, in order:\n\n{transcript}\n\nOriginal summary:\n{summary}\n\n"
              "Problem found by fact-checking:\n{explanation}\n\nCorrected summary:"),
])

correction_chain = correction_prompt | summary_llm | StrOutputParser()


def correct_summary(summary, explanation, chunks_path="chunks.json", keyframes_path="keyframes/keyframes.json"):
    """Runs once, after a summary fails verification -- fixes the specific named
    issue using the full audio+visual context, rather than showing the user a
    flawed answer with a vague warning attached."""
    full_context = build_full_context(chunks_path, keyframes_path)
    return correction_chain.invoke({
        "transcript": full_context,
        "summary": summary,
        "explanation": explanation,
    })


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        query = input("Ask a question about the video: ")
        answer = chain.invoke(query)
        print(f"\nAnswer:\n{answer}\n")