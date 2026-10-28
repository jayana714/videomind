"""Hosted demo: ask questions about pre-processed videos.

The full app (app.py) downloads and analyzes any YouTube link, which needs more memory
than a free host offers and gets blocked by YouTube from cloud IPs. This version ships
with a video already processed, so visitors can try the real question-answering
pipeline -- routing, retrieval, generation, fact-checking -- with nothing to wait for.

It runs as a FastAPI service: the Gradio chat UI is mounted at "/", with a JSON API
(POST /api/ask) and a health check (GET /health) alongside it."""

import datetime
import os
from typing import Optional

# Must be set BEFORE chain.py is imported -- it picks its lightweight code path at import.
os.environ["DEMO_MODE"] = "1"

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

import demo_store
import chain as chain_module
import graph as graph_module
from qa import respond

VIDEOS = demo_store.list_videos()
try:
    demo_store.warm_up()  # load the embedding model now, not inside the first user request
except Exception as e:
    print(f"[Warning: embedding model warm-up failed, first question may fail too: {e}]")
# Every question spends the owner's API credits, so cap total questions per day
# (resets at midnight server time). Pair this with a spend limit in the Anthropic console.
DAILY_QUESTION_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", "40"))
_usage = {"day": None, "count": 0}


def select_video(video_id):
    demo_store.set_video(video_id)
    chain_module.refresh()
    graph_module.refresh_visual_retriever()

    meta = dict(VIDEOS)[video_id]
    status = (
        f"✅ **{meta['name']}** is loaded — [watch the original video]({meta['url']}). "
        f"If you switch videos, click the chat's clear button first."
    )
    questions = meta.get("questions") or []
    suggestions = "**Try asking:**\n" + "\n".join(f"- {q}" for q in questions) if questions else ""
    return status, suggestions


def limited_respond(message, history):
    today = datetime.date.today()
    if _usage["day"] != today:
        _usage["day"], _usage["count"] = today, 0
    if _usage["count"] >= DAILY_QUESTION_LIMIT:
        return ("⚠️ This demo has hit its daily question limit to keep API costs in check. "
                "Please come back tomorrow, or clone the repo and run it with your own key.")
    _usage["count"] += 1
    try:
        return respond(message, history)
    except Exception as e:
        return f"⚠️ Something went wrong answering that: {e}"


# ---------------------------------------------------------------- FastAPI service

api = FastAPI(title="VideoMind", description="Ask questions about a video's audio and on-screen content.")


class AskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None


@api.get("/health")
def health():
    """Liveness check used by the host (Render) and handy for monitoring."""
    return {
        "status": "ok",
        "videos": [vid for vid, _ in VIDEOS],
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "questions_today": _usage["count"] if _usage["day"] == datetime.date.today() else 0,
        "daily_limit": DAILY_QUESTION_LIMIT,
    }


@api.post("/api/ask")
def ask(req: AskRequest):
    if req.video_id and req.video_id != demo_store._state["video_id"]:
        if req.video_id not in dict(VIDEOS):
            return {"error": f"Unknown video_id. Choose one of: {[v for v, _ in VIDEOS]}"}
        select_video(req.video_id)
    return {"video_id": demo_store._state["video_id"], "answer": limited_respond(req.question, [])}


# ---------------------------------------------------------------- Gradio UI

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎬 VideoMind — live demo")
    gr.Markdown(
        "Ask questions about what was **said** and what was **shown on screen** in a video, "
        "instead of watching all of it. Every answer is fact-checked against the source "
        "before you see it, and auto-corrected if the check flags a problem. "
        "This hosted demo uses videos that were already processed; to "
        "analyze any YouTube link, run the full app from the "
        "[GitHub repo](https://github.com/jayana714/videomind).\n\n"
        "*The first load can take about a minute while the free server wakes up.*"
    )

    if not VIDEOS:
        gr.Markdown("⚠️ No demo videos found. Run `python build_demo_data.py` and redeploy.")
    else:
        choices = [(meta["name"], vid) for vid, meta in VIDEOS]
        video_dd = gr.Dropdown(choices=choices, value=VIDEOS[0][0], label="Choose a video", interactive=True)
        status = gr.Markdown()
        suggestions = gr.Markdown()
        video_dd.change(select_video, video_dd, [status, suggestions])
        demo.load(select_video, video_dd, [status, suggestions])

        gr.ChatInterface(
            fn=limited_respond,
            chatbot=gr.Chatbot(height=420, placeholder="Pick a video above, then ask a question."),
        )

if VIDEOS:
    select_video(VIDEOS[0][0])  # so the very first question works before anyone touches the dropdown

# The deployable app: FastAPI service with the Gradio UI mounted at "/".
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))