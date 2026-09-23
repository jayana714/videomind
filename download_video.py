import os
import glob
import yt_dlp

def download_video(url, output_path="video"):
    # Delete any previously downloaded file with this name FIRST -- otherwise
    # yt-dlp silently skips downloading if a file already exists, silently
    # reusing an OLD video's file for a brand NEW video url. This is exactly what
    # caused keyframes from a previous video to persist after processing a new one.
    for existing_file in glob.glob(f"{output_path}.*"):
        os.remove(existing_file)

    """Downloads the actual video (not just audio) for keyframe extraction in v2.
    720p balances readability of on-screen text/UI/tables (what the vision model
    actually needs to describe frames accurately) against file size -- 480p turned
    out to be too low-resolution for reliably reading small text and table values,
    which was the likely root cause of several vision hallucinations."""
    ydl_opts = {
        "format": "best[height<=720]/best",
        "outtmpl": f"{output_path}.%(ext)s",
        # Same SABR workaround from download_audio.py -- YouTube's streaming
        # restriction affects video downloads too, not just audio-only ones.
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Done. Saved video file.")

if __name__ == "__main__":
    video_url = input("Paste a video link: ")
    download_video(video_url)