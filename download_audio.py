import glob
import os
import yt_dlp

def download_audio(url, output_path="audio"):
    # Same fix as download_video.py -- delete any old file first so yt-dlp can't
    # silently skip downloading and reuse stale audio from a previous video.
    for existing_file in glob.glob(f"{output_path}.*"):
        os.remove(existing_file)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # Workaround for YouTube's SABR streaming rollout, which currently breaks
        # the default web-based clients for many videos. The android client has
        # generally been more reliable against this restriction (as of late 2026) --
        # see https://github.com/yt-dlp/yt-dlp/issues/12482 for the ongoing situation.
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        # NOTE: cookies-from-browser was tried here to dodge YouTube's bot-check,
        # but it disables the "android" client (incompatible with cookies), which
        # left only the SABR-restricted "web" client's format 18 -- and THAT format
        # 403s when cookies are attached. Reverted -- back to SABR workaround only.
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
    print("Done. Saved as audio.mp3")

if __name__ == "__main__":
    video_url = input("Paste a video link: ")
    download_audio(video_url)