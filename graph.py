import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import chain as chain_module  # reuse the existing llm, audio retriever, embeddings, format_docs

if not chain_module.DEMO_MODE:
    from langchain_chroma import Chroma
else:
    import demo_store

load_dotenv()


class VideoMindState(TypedDict):
    question: str
    route: List[str]
    audio_context: str
    visual_context: str
    visual_paths: List[str]
    answer: str
    # Verification results: "yes" (checked, grounded), "corrected" (flagged, then fixed),
    # or "unchecked" (the checker call itself failed -- we say so rather than pretend).
    verification: str
    verification_note: str
    verification_context: str  # the evidence the checker saw; reused by the correction step
    original_question: str  # exactly what the user typed, before any follow-up rewriting


# --- Routing: decide what kind of information this question actually needs ---

ROUTE_SYSTEM_PROMPT = """Decide what kind of information is needed to answer a question about a
video that has both spoken content (audio/transcript) and on-screen visual content (keyframe
descriptions of the UI, maps, tables, etc. shown).

Respond with exactly one word:
- "audio" -- the question is about what was said, explained, or discussed
- "visual" -- the question is about what's shown on screen (what something looks like, colors,
  layout, table contents, a specific dialog or panel)
- "both" -- genuinely needs both, e.g. "walk me through what they did and show me what it looked like"
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTE_SYSTEM_PROMPT),
    ("human", "{question}"),
])
route_chain = route_prompt | chain_module.classifier_llm | StrOutputParser()


def build_visual_retriever():
    if chain_module.DEMO_MODE:
        return demo_store.visual_retriever(k=3)
    vectorstore = Chroma(
        collection_name="video_keyframes",
        embedding_function=chain_module.embeddings,
        persist_directory="./chroma_db",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


visual_retriever = build_visual_retriever()


def refresh_visual_retriever():
    """Same idea as chain.py's refresh() -- call after storing a new video's
    keyframes so this doesn't keep querying stale data."""
    global visual_retriever
    visual_retriever = build_visual_retriever()


def classify_route(state: VideoMindState) -> dict:
    result = route_chain.invoke({"question": state["question"]}).strip().lower()
    if "both" in result:
        route = ["audio", "visual"]
    elif "visual" in result:
        # Visual questions ALSO pull the spoken content. A single frame can show a
        # setting's default value before the narrator changes it ("match option" read
        # "Intersect" at 00:56, but the narration later switches it to "closest"); frames
        # alone can't tell you which value was actually used. Audio supplies that context.
        route = ["audio", "visual"]
    else:
        route = ["audio"]
    print(f"[Route: {route}]")
    return {"route": route}


def route_decision(state: VideoMindState):
    # Returning a list here tells LangGraph to fan out to ALL matching branches
    # (e.g. both retrieve_audio and retrieve_visual run when route is "both")
    return state["route"]


# --- Retrieval: pull from whichever source(s) the router picked ---

def retrieve_audio(state: VideoMindState) -> dict:
    docs = chain_module.retriever.invoke(state["question"])
    return {"audio_context": chain_module.format_docs(docs)}


def retrieve_visual(state: VideoMindState) -> dict:
    docs = visual_retriever.invoke(state["question"])
    blocks, paths = [], []
    for doc in docs:
        ts = chain_module.format_timestamp(doc.metadata["timestamp"])
        blocks.append(f"[Frame at {ts}]\n{doc.page_content.strip()}")
        paths.append(doc.metadata["path"])
    return {"visual_context": "\n\n".join(blocks), "visual_paths": paths}


# --- Generation: answer using whichever context(s) were actually retrieved ---

ANSWER_SYSTEM_PROMPT = """Answer the question about a video using the context provided below.
You may be given spoken-content excerpts (what was said), visual descriptions (what was shown
on screen), or both -- use whichever is relevant and actually available. Cite approximate
timestamps. If the available context doesn't actually answer the question, say so honestly
instead of guessing.

The sources come from different moments in the video. A setting or value seen on screen early
on may just be a default that the speaker changes later. When sources conflict, prefer what the
narration says was actually chosen or used, mention the earlier on-screen value only if it helps,
and say which timestamp each part comes from.

If the user's exact wording assumes something the material contradicts (for example, asking why
the narrator chose one option when the video shows a different one), say so explicitly first, then
give the correct information. Do not quietly answer a corrected version of their question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    ("human", "Spoken content (audio):\n{audio_context}\n\n"
              "Visual content (on-screen):\n{visual_context}\n\nQuestion: {question}\n\n"
              "The user's exact original wording (the question above may be a rewrite): {original_question}"),
])
answer_chain = answer_prompt | chain_module.llm | StrOutputParser()


def generate_answer(state: VideoMindState) -> dict:
    answer = answer_chain.invoke({
        "audio_context": state.get("audio_context") or "(not retrieved -- not needed for this question)",
        "visual_context": state.get("visual_context") or "(not retrieved -- not needed for this question)",
        "question": state["question"],
        "original_question": state.get("original_question") or state["question"],
    })
    return {"answer": answer}


# --- Verification: fact-check EVERY specific answer ---

# Videos whose COMPLETE audio+visual context fits under this many characters (~8k tokens,
# roughly a cent of Haiku input) are verified against the WHOLE video, so a retrieval miss
# can't slip through. Longer videos fall back to the retrieved context PLUS fresh evidence
# retrieved with the ANSWER itself as the query (see _verification_context).
FULL_CONTEXT_MAX_CHARS = 32000

VERIFY_SYSTEM_PROMPT = """You are fact-checking an answer about a video against source material
from that video: spoken-content excerpts (what was said) and/or on-screen frame descriptions
(what was shown), each with timestamps. {scope_note}

Check:
1. Content: is every specific claim in the answer supported by the material?
2. Timestamps: does each cited timestamp match the excerpt or frame the claim came from?
3. Changes over time: if the material shows a value changing (e.g. a setting visible at its
   default early on, then changed later), the answer must reflect the value actually used or
   final -- an answer that only reports the earlier value is wrong.
4. Refusals: an answer saying the video doesn't cover something is grounded ONLY if the
   material truly lacks that information. If the material contains the answer, mark it "no".
5. False premises: compare the answer with the user's EXACT original wording. If that wording
   assumes something the material contradicts, the answer must say so explicitly. An answer that
   silently responds to a corrected version of the question is wrong -- mark it "no".

Do not use outside knowledge to judge claims -- only the provided material.

Respond in exactly this format:
Grounded: yes or no
Explanation: one or two sentences. If "no", name the specific claim, timestamp, or omission that is wrong."""

verify_prompt = ChatPromptTemplate.from_messages([
    ("system", VERIFY_SYSTEM_PROMPT),
    ("human", "Source material:\n{context}\n\nQuestion as processed: {question}\n\n"
              "User's exact original wording: {original_question}\n\nGenerated answer:\n{answer}"),
])
verify_chain = verify_prompt | chain_module.llm | StrOutputParser()

CORRECT_SYSTEM_PROMPT = """You wrote an answer about a video, and a fact-checker found a specific
problem with it. Fix ONLY the problem described, using only the provided source material. Keep
everything that was already correct. If the material cannot support an answer to the question,
say so plainly instead of guessing. Output the complete corrected answer, not just the fix."""

correct_prompt = ChatPromptTemplate.from_messages([
    ("system", CORRECT_SYSTEM_PROMPT),
    ("human", "Source material:\n{context}\n\nQuestion as processed: {question}\n\n"
              "User's exact original wording: {original_question}\n\nOriginal answer:\n{answer}\n\n"
              "Problem found by fact-checking:\n{explanation}\n\nCorrected answer:"),
])
correct_chain = correct_prompt | chain_module.llm | StrOutputParser()


def _format_visual_docs(docs):
    return "\n\n".join(
        f"[Frame at {chain_module.format_timestamp(d.metadata['timestamp'])}]\n{d.page_content.strip()}"
        for d in docs
    )


def _verification_context(state):
    """Decides what evidence the fact-checker gets. Returns (context, scope_note).

    The first version of the checker only saw what retrieval had already found, so it could
    pass an answer that retrieval got incomplete (a frame showed a setting's default value,
    the narration later changed it, and only the frame was retrieved). Giving it more than
    the generator saw is what lets it catch that."""
    try:
        full = chain_module.build_full_context()
        if len(full) <= FULL_CONTEXT_MAX_CHARS:
            return full, ("You are given the COMPLETE audio transcript and ALL frame descriptions "
                          "for the whole video, so a claim that appears nowhere in it is unsupported.")
    except Exception as e:
        print(f"[Full-video context unavailable, using retrieved evidence instead: {e}]")

    # Long video: original evidence + fresh evidence retrieved using the ANSWER's own claims.
    extra_audio = chain_module.format_docs(chain_module.retriever.invoke(state["answer"]))
    extra_visual = _format_visual_docs(visual_retriever.invoke(state["answer"]))
    context = (
        f"EVIDENCE THE ANSWER WAS BUILT FROM\nSpoken:\n{state.get('audio_context') or '(none)'}\n\n"
        f"On-screen:\n{state.get('visual_context') or '(none)'}\n\n"
        f"ADDITIONAL PASSAGES RETRIEVED USING THE ANSWER'S OWN CLAIMS\nSpoken:\n{extra_audio}\n\n"
        f"On-screen:\n{extra_visual}"
    )
    return context, ("You are given the evidence the answer was built from, plus additional passages "
                     "retrieved using the answer's own claims. This is part of a longer video, so "
                     "absence from these passages is weaker evidence than contradiction by them.")


def verify_answer(state: VideoMindState) -> dict:
    try:
        context, scope_note = _verification_context(state)
        result = verify_chain.invoke({
            "context": context, "scope_note": scope_note,
            "question": state["question"], "answer": state["answer"],
            "original_question": state.get("original_question") or state["question"],
        })
    except Exception as e:
        # A failed checker must never be reported as a passed check.
        print(f"[Verification could not run: {e}]")
        return {"verification": "unchecked", "verification_note": str(e), "verification_context": ""}

    grounded = "yes" in result.lower().split("\n")[0]
    explanation = result.split("Explanation:", 1)[-1].strip()
    print(f"[Verification: {'grounded' if grounded else 'FLAGGED -- ' + explanation}]")
    return {
        "verification": "yes" if grounded else "flagged",
        "verification_note": explanation,
        "verification_context": context,
    }


def after_verification(state: VideoMindState):
    return "correct_answer" if state["verification"] == "flagged" else END


def correct_answer(state: VideoMindState) -> dict:
    # One correction attempt only, same cap as the summary path: re-verifying again
    # would double the cost for diminishing returns, so we trust the targeted fix.
    try:
        fixed = correct_chain.invoke({
            "context": state["verification_context"],
            "question": state["question"],
            "original_question": state.get("original_question") or state["question"],
            "answer": state["answer"],
            "explanation": state["verification_note"],
        })
    except Exception as e:
        print(f"[Correction failed, keeping original answer: {e}]")
        return {"verification": "unchecked", "verification_note": str(e)}
    return {"answer": fixed, "verification": "corrected"}


# --- Wiring the graph together ---

graph_builder = StateGraph(VideoMindState)
graph_builder.add_node("classify_route", classify_route)
graph_builder.add_node("retrieve_audio", retrieve_audio)
graph_builder.add_node("retrieve_visual", retrieve_visual)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("verify_answer", verify_answer)
graph_builder.add_node("correct_answer", correct_answer)

graph_builder.add_edge(START, "classify_route")
graph_builder.add_conditional_edges(
    "classify_route",
    route_decision,
    {"audio": "retrieve_audio", "visual": "retrieve_visual"},
)
graph_builder.add_edge("retrieve_audio", "generate_answer")
graph_builder.add_edge("retrieve_visual", "generate_answer")
graph_builder.add_edge("generate_answer", "verify_answer")
graph_builder.add_conditional_edges(
    "verify_answer",
    after_verification,
    {"correct_answer": "correct_answer", END: END},
)
graph_builder.add_edge("correct_answer", END)

graph = graph_builder.compile()


def ask_multimodal_verified(question, original_question=None):
    """Full result including the fact-check outcome, so the UI can show an honest badge.
    `original_question` is what the user actually typed; `question` may be a rewrite of it."""
    result = graph.invoke({
        "question": question,
        "original_question": original_question or question,
        "route": [],
        "audio_context": "",
        "visual_context": "",
        "visual_paths": [],
        "answer": "",
        "verification": "unchecked",
        "verification_note": "",
        "verification_context": "",
    })
    return result["answer"], result.get("visual_paths", []), result["verification"]


def ask_multimodal(question):
    answer, paths, _ = ask_multimodal_verified(question)
    return answer, paths


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        q = input("Ask a question about the video (audio + visual): ")
        answer, paths = ask_multimodal(q)
        print(f"\nAnswer:\n{answer}\n")
        if paths:
            print(f"Referenced frames: {paths}")