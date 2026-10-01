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
        # NOTE: cookies-from-browser was tried here to dodge YouTube's bot-check,
        # but it disables the "android" client (incompatible with cookies), which
        # left only the SABR-restricted "web" client's format 18 -- and THAT format
        # 403s when cookies are attached. Net result: cookies broke normal single-
        # video downloads without reliably fixing the eval script's bot-check issue.
        # Reverted -- back to SABR workaround only.
    }
    # Opt-in only: a cookies.txt file (exported once via a browser extension) is
    # attached ONLY if it actually exists in the working directory. Normal single-
    # video usage never has this file, so it's completely unaffected -- this exists
    # purely so run_evals.py can process several videos back-to-back without
    # tripping YouTube's bot detection, without touching the app's default behavior.
    if os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Done. Saved video file.")

if __name__ == "__main__":
    video_url = input("Paste a video link: ")
    download_video(video_url)