import os
import json
import math
from pydub import AudioSegment
from groq import Groq

# Keeps each piece safely under Groq's free-tier 25MB upload cap, regardless
# of how long the original video is.
CHUNK_MINUTES = 20


def transcribe_audio_groq(audio_path="audio.mp3", progress_callback=None):
    """Transcribes audio via Groq's hosted Whisper API -- roughly 300x faster
    than running Whisper locally on a laptop CPU. Splits long audio into
    chunks first since Groq's free tier caps uploads at 25MB per request."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = CHUNK_MINUTES * 60 * 1000
    num_chunks = math.ceil(total_ms / chunk_ms) if total_ms else 1

    all_segments = []
    for i in range(num_chunks):
        start_ms = i * chunk_ms
        end_ms = min(start_ms + chunk_ms, total_ms)
        chunk_path = f"_groq_chunk_{i}.mp3"
        # 64k mono is plenty for speech recognition and keeps file size small
        audio[start_ms:end_ms].export(chunk_path, format="mp3", bitrate="64k")

        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(chunk_path, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )

        # Shift each chunk's timestamps forward so they line up with the full video
        offset_seconds = start_ms / 1000
        segments = result.segments if hasattr(result, "segments") else result["segments"]
        for seg in segments:
            all_segments.append({
                "start": seg["start"] + offset_seconds,
                "end": seg["end"] + offset_seconds,
                "text": seg["text"],
            })

        os.remove(chunk_path)

        if progress_callback:
            progress_callback(end_ms / 1000, total_ms / 1000)

    with open("transcript.json", "w") as f:
        json.dump(all_segments, f, indent=2)
    print(f"Saved {len(all_segments)} segments to transcript.json (via Groq)")

    return all_segments


if __name__ == "__main__":
    transcribe_audio_groq()