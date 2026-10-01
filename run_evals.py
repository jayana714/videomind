import json
import os
import re
import shutil
import time

from app import respond, process_video
from store_chunks import store_chunks
from store_keyframes import store_keyframes
import chain as chain_module
import graph as graph_module

CACHE_DIR = "eval_cache"


def extract_video_id(url):
    """Pulls the YouTube video ID out of a normal watch URL or a youtu.be short
    link, so each video gets a stable, unique cache folder name."""
    match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    return match.group(1) if match else url


def is_cached(video_id):
    """A video counts as cached only if every file process_video() would have
    produced is actually present -- a partial cache (e.g. from an interrupted
    run) is treated as no cache at all, so we never silently load broken data."""
    cache_path = os.path.join(CACHE_DIR, video_id)
    required = ["audio.mp3", "video.mp4", "chunks.json",
                os.path.join("keyframes", "keyframes.json")]
    return all(os.path.exists(os.path.join(cache_path, f)) for f in required)


def load_from_cache(video_id):
    """Copies a previously cached video's files into the working directory and
    re-runs just the storage/retriever-refresh steps -- no download, no
    transcription, no vision calls, no YouTube contact at all. This is what
    makes repeat eval runs fast AND immune to YouTube's bot detection."""
    cache_path = os.path.join(CACHE_DIR, video_id)

    shutil.copy(os.path.join(cache_path, "audio.mp3"), "audio.mp3")
    shutil.copy(os.path.join(cache_path, "video.mp4"), "video.mp4")
    shutil.copy(os.path.join(cache_path, "chunks.json"), "chunks.json")

    if os.path.exists("keyframes"):
        shutil.rmtree("keyframes")
    shutil.copytree(os.path.join(cache_path, "keyframes"), "keyframes")

    store_chunks()
    chain_module.refresh()
    store_keyframes()
    graph_module.refresh_visual_retriever()


def save_to_cache(video_id):
    """After a fresh process_video() run succeeds, copies its output files into
    the cache so every future eval run of this same video skips YouTube
    entirely."""
    cache_path = os.path.join(CACHE_DIR, video_id)
    os.makedirs(cache_path, exist_ok=True)

    shutil.copy("audio.mp3", os.path.join(cache_path, "audio.mp3"))
    shutil.copy("video.mp4", os.path.join(cache_path, "video.mp4"))
    shutil.copy("chunks.json", os.path.join(cache_path, "chunks.json"))

    dest_keyframes = os.path.join(cache_path, "keyframes")
    if os.path.exists(dest_keyframes):
        shutil.rmtree(dest_keyframes)
    shutil.copytree("keyframes", dest_keyframes)


def process_video_cached(url):
    """Drop-in replacement for process_video() that checks the local cache
    first. Returns the same (status, examples) shape process_video() does."""
    video_id = extract_video_id(url)

    if is_cached(video_id):
        print(f"  📦 Using cached files for {video_id} -- skipping download entirely.")
        load_from_cache(video_id)
        return "✅ Ready! (loaded from cache)", ""

    print(f"  🌐 No cache for {video_id} yet -- running the full pipeline once.")
    status, examples = process_video(url)
    if not (status.startswith("❌") or status.startswith("⚠️")):
        save_to_cache(video_id)
        print(f"  💾 Cached {video_id} for future eval runs.")
    return status, examples


def run_evals(eval_path="eval_cases.json"):
    """Full automated regression suite: for each video, processes it once (via
    the cache -- see process_video_cached above), then runs its test questions
    and checks them against known-correct answers. This is the difference
    between spot-testing by hand and having an actual measured pass rate."""
    with open(eval_path) as f:
        video_groups = json.load(f)

    total_passed = 0
    total_failed = 0
    results_by_video = []

    for group_index, group in enumerate(video_groups):
        video_name = group["video_name"]
        video_url = group["video_url"]
        cases = group["cases"]

        if video_url == "REPLACE_ME":
            print(f"\n⚠️  Skipping '{video_name}' -- no video URL set yet.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {video_name}")
        print(f"{'=' * 60}")

        start = time.time()
        status, _ = process_video_cached(video_url)
        elapsed = time.time() - start

        if status.startswith("❌") or status.startswith("⚠️"):
            print(f"⚠️  Could not process this video, skipping its {len(cases)} test case(s): {status}")
            total_failed += len(cases)
            continue

        print(f"{status} (took {elapsed:.0f}s)\n")

        video_passed = 0
        video_failed = 0

        for i, case in enumerate(cases):
            question = case["question"]
            expect_contains = case.get("expect_contains", [])
            expect_not_contains = case.get("expect_not_contains", [])

            answer = respond(question, [])
            answer_lower = answer.lower()

            missing = [kw for kw in expect_contains if kw.lower() not in answer_lower]
            unexpected = [kw for kw in expect_not_contains if kw.lower() in answer_lower]
            ok = not missing and not unexpected

            status_icon = "✅ PASS" if ok else "❌ FAIL"
            print(f"  [{i + 1}/{len(cases)}] {status_icon} — {question}")
            if case.get("note"):
                print(f"     Note: {case['note']}")
            if missing:
                print(f"     Missing expected content: {missing}")
            if unexpected:
                print(f"     Contains unexpected content: {unexpected}")
            if not ok:
                print(f"     Full answer:\n{answer}\n")

            video_passed += ok
            video_failed += not ok

        total_passed += video_passed
        total_failed += video_failed
        results_by_video.append((video_name, video_passed, video_failed))

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS BY VIDEO")
    print(f"{'=' * 60}")
    for name, passed, failed in results_by_video:
        total = passed + failed
        pct = (passed / total * 100) if total else 0
        print(f"  {name}: {passed}/{total} passed ({pct:.0f}%)")

    grand_total = total_passed + total_failed
    grand_pct = (total_passed / grand_total * 100) if grand_total else 0
    print(f"\nOVERALL: {total_passed}/{grand_total} passed ({grand_pct:.0f}%)")

    return total_passed, total_failed


if __name__ == "__main__":
    run_evals()