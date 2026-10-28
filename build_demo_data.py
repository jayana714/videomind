"""Packages already-processed videos (from eval_cache/) into demo_data/ for the hosted demo.

Run this LOCALLY, once, after run_evals.py has cached your videos:

    python build_demo_data.py

For each cached video it writes demo_data/<video_id>/ containing the transcript chunks,
keyframe descriptions, precomputed embeddings, and a meta.json (display name + suggested
questions). The hosted demo then needs no downloading, transcription, or vision calls --
only the Anthropic API for answering. Commit demo_data/ to GitHub (it is just text and
small .npy files, no video or audio)."""

import json
import os
import shutil
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from run_evals import extract_video_id, CACHE_DIR

DEMO_DIR = "demo_data"


def embed(model, texts):
    vecs = model.encode(texts)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs.astype(np.float32)


def main():
    with open("eval_cases.json") as f:
        groups = json.load(f)
    names = {extract_video_id(g["video_url"]): g["video_name"] for g in groups}
    urls = {extract_video_id(g["video_url"]): g["video_url"] for g in groups}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    os.makedirs(DEMO_DIR, exist_ok=True)
    built = 0
    wanted = set(sys.argv[1:])  # optional: python build_demo_data.py PZwa0Mzq-Dk

    for video_id in sorted(os.listdir(CACHE_DIR)):
        if wanted and video_id not in wanted:
            continue
        src = os.path.join(CACHE_DIR, video_id)
        chunks_path = os.path.join(src, "chunks.json")
        if not os.path.exists(chunks_path):
            continue

        dest = os.path.join(DEMO_DIR, video_id)
        os.makedirs(dest, exist_ok=True)

        with open(chunks_path) as f:
            chunks = json.load(f)
        shutil.copy(chunks_path, os.path.join(dest, "chunks.json"))
        np.save(os.path.join(dest, "chunk_vecs.npy"), embed(model, [c["text"] for c in chunks]))

        kf_path = os.path.join(src, "keyframes", "keyframes.json")
        if os.path.exists(kf_path):
            with open(kf_path) as f:
                keyframes = json.load(f)
            shutil.copy(kf_path, os.path.join(dest, "keyframes.json"))
            np.save(os.path.join(dest, "keyframe_vecs.npy"),
                    embed(model, [k["description"] for k in keyframes]))

        # Suggested questions: ask the LLM once, now, so visitors don't pay for it later.
        questions = []
        try:
            from app import generate_example_questions
            questions = generate_example_questions(chunks)
        except Exception as e:
            print(f"  (couldn't generate suggested questions for {video_id}: {e})")

        meta = {
            "name": names.get(video_id, video_id),
            "url": urls.get(video_id, f"https://www.youtube.com/watch?v={video_id}"),
            "questions": questions,
        }
        with open(os.path.join(dest, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ {meta['name']}: {len(chunks)} chunks -> {dest}/")
        built += 1

    print(f"\nBuilt {built} demo video(s) in {DEMO_DIR}/. Commit that folder to GitHub.")


if __name__ == "__main__":
    main()