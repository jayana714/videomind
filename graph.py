import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

import chain as chain_module  # reuse the existing llm, audio retriever, embeddings, format_docs

load_dotenv()


class VideoMindState(TypedDict):
    question: str
    route: List[str]
    audio_context: str
    visual_context: str
    visual_paths: List[str]
    answer: str


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
        route = ["visual"]
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
instead of guessing."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    ("human", "Spoken content (audio):\n{audio_context}\n\n"
              "Visual content (on-screen):\n{visual_context}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | chain_module.llm | StrOutputParser()


def generate_answer(state: VideoMindState) -> dict:
    answer = answer_chain.invoke({
        "audio_context": state.get("audio_context") or "(not retrieved -- not needed for this question)",
        "visual_context": state.get("visual_context") or "(not retrieved -- not needed for this question)",
        "question": state["question"],
    })
    return {"answer": answer}


# --- Wiring the graph together ---

graph_builder = StateGraph(VideoMindState)
graph_builder.add_node("classify_route", classify_route)
graph_builder.add_node("retrieve_audio", retrieve_audio)
graph_builder.add_node("retrieve_visual", retrieve_visual)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "classify_route")
graph_builder.add_conditional_edges(
    "classify_route",
    route_decision,
    {"audio": "retrieve_audio", "visual": "retrieve_visual"},
)
graph_builder.add_edge("retrieve_audio", "generate_answer")
graph_builder.add_edge("retrieve_visual", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


def ask_multimodal(question):
    result = graph.invoke({
        "question": question,
        "route": [],
        "audio_context": "",
        "visual_context": "",
        "visual_paths": [],
        "answer": "",
    })
    return result["answer"], result.get("visual_paths", [])


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        q = input("Ask a question about the video (audio + visual): ")
        answer, paths = ask_multimodal(q)
        print(f"\nAnswer:\n{answer}\n")
        if paths:
            print(f"Referenced frames: {paths}")