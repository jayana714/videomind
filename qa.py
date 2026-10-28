"""Question-answering entry point, shared by the full app (app.py) and the hosted
demo (demo_app.py). Lives in its own module so the demo can import it without
dragging in yt-dlp, Whisper, OpenCV, and the rest of the processing pipeline."""

import chain as chain_module
import graph as graph_module


def _as_text(content):
    """Gradio's message content is a plain string in older versions, but a list of content
    blocks (e.g. [{"type": "text", "text": "..."}]) in newer ones -- the hosted build installs
    the newest. Flattens any of those shapes to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _as_text(content.get("text", content.get("content", "")))
    if isinstance(content, (list, tuple)):
        return "".join(_as_text(part) for part in content)
    return str(content)


def format_history(history):
    """Turns Gradio's chat history into plain text for the condense step.
    Handles both the dict-based and legacy tuple-based history formats."""
    lines = []
    for turn in history:
        if isinstance(turn, dict):
            role, content = turn.get("role"), _as_text(turn.get("content", ""))
            content = content.split("\n\n---\n")[0]  # drop the ✅/⚠️ badge line
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        else:
            user_msg, bot_msg = turn
            lines.append(f"User: {_as_text(user_msg)}")
            if bot_msg:
                lines.append(f"Assistant: {_as_text(bot_msg).split(chr(10)+chr(10)+'---'+chr(10))[0]}")
    return "\n".join(lines)


def respond(message, history):
    message = _as_text(message)
    history_text = format_history(history)
    standalone_query = chain_module.condense_question(message, history_text)
    print(f"[Rewritten query: {standalone_query}]")

    intent = chain_module.classify_intent(standalone_query)
    print(f"[Intent: {intent}]")

    if intent == "summary":
        # Broad questions get the WHOLE transcript, not just 3 retrieved chunks --
        # that's the only way to actually cover a whole video, not a fragment of it.
        # Cache is keyed on the RAW message, not standalone_query -- condense_question
        # can subtly reword even an already-standalone repeat, which would otherwise
        # silently break exact-match caching on a word-for-word repeated question.
        cached = chain_module.get_cached_summary(message)
        if cached:
            answer, verification = cached
            print("[Using cached summary — no API calls made]")
        else:
            answer = chain_module.summarize_video(standalone_query)
            verification = chain_module.verify_summary(answer)
            grounded = "yes" in verification.lower().split("\n")[0]

            if not grounded:
                explanation = verification.split("Explanation:", 1)[-1].strip()
                print(f"[Verification flagged an issue, attempting correction: {explanation}]")
                answer = chain_module.correct_summary(answer, explanation)
                # One correction attempt only -- re-verifying again would double the
                # cost a second time for diminishing returns; trust the targeted fix.

            chain_module.cache_summary(message, answer, verification)

        grounded = "yes" in verification.lower().split("\n")[0]
        if grounded:
            badge = "✅ Verified against full transcript"
        else:
            badge = "✅ Auto-corrected after an initial check flagged an issue"
        if cached:
            badge += " (cached — instant, no cost)"
        return f"{answer}\n\n---\n{badge}"

    # Specific questions go through the multimodal router instead of audio-only
    # retrieval -- this is what makes the visual pipeline usable from the chat itself.
    answer, visual_paths, verification = graph_module.ask_multimodal_verified(standalone_query, original_question=message)
    badge = {
        "yes": "✅ Fact-checked against the video's audio and on-screen content",
        "corrected": "✅ Auto-corrected after the fact-check flagged an issue",
        "unchecked": "⚠️ Fact-check couldn't run for this answer — treat with care",
    }[verification]
    if visual_paths:
        badge += f" ({len(visual_paths)} frame{'s' if len(visual_paths) != 1 else ''} referenced)"
    return f"{answer}\n\n---\n{badge}"