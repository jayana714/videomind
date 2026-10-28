import os
import json
import yt_dlp
import gradio as gr
from dotenv import load_dotenv

from download_audio import download_audio
from download_video import download_video
from chunk_transcript import chunk_segments
from store_chunks import store_chunks
from extract_keyframes import extract_keyframes
from describe_keyframes import describe_all_keyframes
from verify_keyframes import verify_all_keyframes
from store_keyframes import store_keyframes
import chain as chain_module
import graph as graph_module
from qa import respond, format_history  # noqa: F401  (re-exported; run_evals.py imports respond from here)

load_dotenv()

# Use Groq's hosted Whisper (seconds per hour of audio) if a key is set --
# otherwise fall back to local faster-whisper (roughly real-time on CPU).
if os.environ.get("GROQ_API_KEY"):
    from transcribe_groq import transcribe_audio_groq as transcribe_fn
    USING_GROQ = True
else:
    from transcribe import transcribe_audio as transcribe_fn
    USING_GROQ = False

# Local transcription is slow enough that very long videos aren't practical;
# Groq removes that constraint almost entirely, so the cap only applies locally.
MAX_DURATION_MINUTES = 45 if not USING_GROQ else 360


def generate_example_questions(chunks):
    """Ask the LLM to propose a few natural questions based on a sample of THIS
    video's actual content, so suggestions match whatever's currently loaded."""
    sample_text = "\n".join(c["text"] for c in chunks[:6])
    prompt = (
        "Here are the first few excerpts from a video transcript. Suggest exactly 4 short, "
        "natural questions a viewer might ask about this video. Return ONLY the 4 questions, "
        "one per line, no numbering, no extra text.\n\nExcerpts:\n" + sample_text
    )
    response = chain_module.llm.invoke(prompt)
    questions = [q.strip("-• ").strip() for q in response.content.split("\n") if q.strip()]
    return questions[:4]


def get_video_duration(url):
    """Check how long the video is before committing to downloading/transcribing it."""
    ydl_opts = {
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        # Reverted cookies-from-browser -- it broke normal downloads with a 403
        # (see download_video.py for the full explanation). Back to SABR-only.
    }
    # Same opt-in cookies.txt support as download_audio.py/download_video.py --
    # only active if the file exists (eval runs), no effect on normal app usage.
    if os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("duration")  # seconds; may be None for some live streams


def process_video(url, progress=gr.Progress()):
    """Runs the FULL v1 + v2 pipeline on one video: audio download -> transcribe ->
    chunk -> store, AND video download -> keyframes -> describe -> verify -> store.
    Refreshes BOTH retrievers at the end so audio and visual answers always come
    from the same video -- this is the fix for the audio/visual mismatch found
    when the two pipelines were run independently."""
    if not url or not url.strip():
        return "⚠️ Paste a video link first.", ""

    try:
        progress(0.02, desc="Checking video length...")
        duration = get_video_duration(url)
        if duration and duration > MAX_DURATION_MINUTES * 60:
            minutes = round(duration / 60)
            return (
                f"⚠️ This video is about {minutes} minutes long. To keep processing time "
                f"reasonable, VideoMind currently supports videos under {MAX_DURATION_MINUTES} "
                f"minutes{'​ with local transcription' if not USING_GROQ else ''}. Try a shorter "
                f"video, or raise MAX_DURATION_MINUTES in app.py if you're willing to wait longer."
            ), ""

        # --- Audio pipeline (v1, Steps 1-6) ---
        progress(0.05, desc="Downloading audio...")
        download_audio(url)

        def report_transcription_progress(current_seconds, total_seconds):
            fraction = current_seconds / total_seconds if total_seconds else 0
            progress(0.10 + fraction * 0.30, desc=f"Transcribing... {int(fraction * 100)}%")

        transcribe_desc = "Transcribing via Groq (fast)..." if USING_GROQ else "Transcribing — this is the slow part, hang tight..."
        progress(0.10, desc=transcribe_desc)
        segments = transcribe_fn("audio.mp3", progress_callback=report_transcription_progress)

        progress(0.42, desc="Chunking transcript...")
        chunks = chunk_segments(segments)
        with open("chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)

        progress(0.46, desc="Embedding and storing audio in the vector database...")
        store_chunks()
        chain_module.refresh()

        # --- Visual pipeline (v2, Steps 1-3) ---
        progress(0.50, desc="Downloading video for keyframes...")
        download_video(url)

        progress(0.58, desc="Extracting keyframes...")
        keyframes = extract_keyframes()

        progress(0.65, desc="Describing what's on screen (this takes a bit, one call per frame)...")
        describe_all_keyframes()

        progress(0.85, desc="Fact-checking keyframe descriptions...")
        verify_all_keyframes()

        progress(0.93, desc="Embedding and storing visuals in the vector database...")
        store_keyframes()
        graph_module.refresh_visual_retriever()

        progress(0.97, desc="Coming up with example questions...")
        try:
            questions = generate_example_questions(chunks)
            suggestions_md = "**Try asking:**\n" + "\n".join(f"- {q}" for q in questions)
        except Exception:
            suggestions_md = ""  # non-critical -- don't fail the whole run over this

        progress(1.0, desc="Done!")
        return (
            f"✅ Ready! Processed {len(chunks)} audio chunks and {len(keyframes)} keyframes "
            f"from the video. Ask a question below.",
            suggestions_md,
        )

    except Exception as e:
        return f"❌ Something went wrong: {e}", ""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎬 VideoMind")
    gr.Markdown(
        "Paste a video link below and process it once. After that, ask it anything instead of "
        "watching the whole video — including what was **said** and what was **shown on screen**. "
        "Answers are generated only from the actual video content, and every answer is "
        "automatically fact-checked before you see it. (Processing now includes downloading and "
        "analyzing video frames, not just audio, so it takes a bit longer than before.)"
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="Video URL",
            placeholder="Paste a YouTube (or other) video link here...",
            scale=4,
        )
        process_btn = gr.Button("Process video", variant="primary", scale=1)

    status = gr.Markdown()
    suggestions = gr.Markdown()
    process_btn.click(fn=process_video, inputs=url_input, outputs=[status, suggestions])

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=420, placeholder="Process a video above, then ask a question here."),
    )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        # Render (and most PaaS hosts) assign the port via $PORT and expect the
        # app to bind on 0.0.0.0, not Gradio's local-only default -- without this
        # the deployed service never becomes reachable even though it "runs".
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))