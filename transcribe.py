from faster_whisper import WhisperModel
import json

def transcribe_audio(audio_path="audio.mp3", model_size="small", progress_callback=None):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # vad_filter skips silence/non-speech stretches -- meaningful speedup on real
    # videos with pauses, with no accuracy cost on the parts that are actually spoken
    segments, info = model.transcribe(audio_path, vad_filter=True)

    print(f"Detected language: {info.language}")
    print("-" * 40)

    results = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        # Report real progress (how far into the audio we are) instead of a guess
        if progress_callback and info.duration:
            progress_callback(segment.end, info.duration)

    # Save raw segments to disk
    with open("transcript.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} segments to transcript.json")

    return results

if __name__ == "__main__":
    transcribe_audio()