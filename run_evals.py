import json
import time
from app import respond, process_video

def run_evals(eval_path="eval_cases.json"):
    """Full automated regression suite: for each video, processes it (download,
    transcribe, extract keyframes, etc. -- the same pipeline a real user triggers
    by clicking 'Process video'), then runs its test questions and checks them
    against known-correct answers. This is the difference between spot-testing
    by hand and having an actual measured pass rate you can point to."""
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

        if group_index > 0:
            wait_seconds = 30
            print(f"\nWaiting {wait_seconds}s before the next video "
                  f"(rapid back-to-back requests can trigger YouTube's bot detection)...")
            time.sleep(wait_seconds)

        print(f"\n{'=' * 60}")
        print(f"Processing: {video_name}")
        print(f"{'=' * 60}")

        start = time.time()
        status, _ = process_video(video_url)
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