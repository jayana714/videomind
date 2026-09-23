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
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Done. Saved as audio.mp3")

if __name__ == "__main__":
    video_url = input("Paste a video link: ")
    download_audio(video_url)