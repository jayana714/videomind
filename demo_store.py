"""Lightweight vector search for the hosted demo.

The full app embeds with PyTorch (sentence-transformers) and stores vectors in Chroma.
That needs well over 512 MB of RAM -- too much for a free host. In demo mode the
videos are processed ahead of time (build_demo_data.py), their vectors are saved as
.npy files, and at runtime only the user's QUESTION needs embedding. We do that with
the same all-MiniLM-L6-v2 model through ONNX (fastembed), so retrieval behaves the
same as the full app, without PyTorch. With a few hundred vectors per video, a plain
numpy dot product is all the "vector database" this needs.
"""

import json
import os
import shutil

import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

DEMO_DIR = "demo_data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = os.environ.get("FASTEMBED_CACHE", "models")

_model = None
_state = {
    "video_id": None,
    "chunks": [], "chunk_vecs": None,
    "keyframes": [], "keyframe_vecs": None,
}


def _embed_query(text):
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL_NAME, cache_dir=MODEL_CACHE)
    vec = np.array(list(_model.embed([text]))[0], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)


def warm_up():
    """Loads (and, the first time, downloads) the embedding model at startup instead of on the
    first question. Downloading inside a Gradio request fails with "type object 'tqdm' has no
    attribute '_lock'", because Gradio patches tqdm while handling requests. In Docker the model
    is already baked into the image, so this is just a fast load."""
    _embed_query("warm up")


def list_videos():
    """Returns [(video_id, meta_dict)] for every pre-processed video in demo_data/."""
    videos = []
    if not os.path.isdir(DEMO_DIR):
        return videos
    for video_id in sorted(os.listdir(DEMO_DIR)):
        meta_path = os.path.join(DEMO_DIR, video_id, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                videos.append((video_id, json.load(f)))
    return videos


def set_video(video_id):
    """Makes `video_id` the active video. Also writes its chunks/keyframes to the
    working directory, because the full-video summary + verification code in chain.py
    reads them from chunks.json and keyframes/keyframes.json."""
    folder = os.path.join(DEMO_DIR, video_id)

    with open(os.path.join(folder, "chunks.json")) as f:
        chunks = json.load(f)
    chunk_vecs = np.load(os.path.join(folder, "chunk_vecs.npy"))

    keyframes, keyframe_vecs = [], None
    kf_path = os.path.join(folder, "keyframes.json")
    if os.path.exists(kf_path):
        with open(kf_path) as f:
            keyframes = json.load(f)
        keyframe_vecs = np.load(os.path.join(folder, "keyframe_vecs.npy"))

    _state.update(
        video_id=video_id,
        chunks=chunks, chunk_vecs=chunk_vecs,
        keyframes=keyframes, keyframe_vecs=keyframe_vecs,
    )

    shutil.copy(os.path.join(folder, "chunks.json"), "chunks.json")
    if os.path.exists("keyframes"):
        shutil.rmtree("keyframes")  # never leave the previous video's frames behind
    if keyframes:
        os.makedirs("keyframes", exist_ok=True)
        shutil.copy(kf_path, os.path.join("keyframes", "keyframes.json"))


def _top_k(vecs, query, k):
    if vecs is None or len(vecs) == 0:
        return []
    scores = vecs @ _embed_query(query)
    return list(np.argsort(-scores)[:k])


def audio_retriever(k=8):
    def search(query):
        return [
            Document(
                page_content=_state["chunks"][i]["text"],
                metadata={"start": _state["chunks"][i]["start"], "end": _state["chunks"][i]["end"]},
            )
            for i in _top_k(_state["chunk_vecs"], query, k)
        ]
    return RunnableLambda(search)


def visual_retriever(k=3):
    def search(query):
        return [
            Document(
                page_content=_state["keyframes"][i]["description"],
                metadata={"timestamp": _state["keyframes"][i]["timestamp"],
                          "path": _state["keyframes"][i].get("path", "")},
            )
            for i in _top_k(_state["keyframe_vecs"], query, k)
        ]
    return RunnableLambda(search)