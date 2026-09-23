import os
import cv2
import json

def compute_frame_diff(frame1, frame2, pixel_threshold=25):
    """Returns the FRACTION of pixels that changed significantly between two frames,
    rather than the average change across the whole frame. A whole-frame average
    gets diluted by large static backgrounds (e.g. a mostly-white spreadsheet), so
    a real but localized change -- a dialog box, a formatted cell range -- can fail
    to move the average enough to register. Counting what fraction of pixels
    actually changed is far more sensitive to exactly that kind of change."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    changed_pixels = (diff > pixel_threshold).sum()
    return changed_pixels / diff.size


def extract_keyframes(video_path="video.mp4", output_dir="keyframes",
                       check_interval_seconds=2, change_threshold=0.03,
                       min_gap_seconds=10, max_keyframes=40):
    """Extracts keyframes based on ACTUAL on-screen changes, not a fixed schedule --
    a candidate frame is checked every check_interval_seconds, and saved only if it
    looks meaningfully different from the last saved keyframe (a new dialog, panel,
    or result appearing). This catches short-lived but important moments that fixed
    sampling would miss between checkpoints.

    min_gap_seconds prevents saving near-duplicate frames back to back during a
    flicker or transition. max_keyframes is a safety valve for unusually high-motion
    video (e.g. constant scrolling) so cost and processing time stay predictable
    even if the video changes far more often than a normal tutorial would."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0 or not total_frames or total_frames <= 0:
        raise ValueError("Could not read FPS/frame count from video -- is the file valid?")

    duration_seconds = total_frames / fps
    check_interval_frames = max(1, int(fps * check_interval_seconds))
    min_gap_frames = max(1, int(fps * min_gap_seconds))

    keyframes = []
    last_saved_frame = None
    last_saved_frame_count = -min_gap_frames  # allows saving immediately at frame 0
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % check_interval_frames == 0:
            is_first = last_saved_frame is None
            enough_gap = (frame_count - last_saved_frame_count) >= min_gap_frames

            if is_first:
                should_save = True
            elif enough_gap:
                should_save = compute_frame_diff(frame, last_saved_frame) > change_threshold
            else:
                should_save = False

            if should_save:
                timestamp = frame_count / fps
                filename = f"{output_dir}/frame_{saved_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                keyframes.append({"path": filename, "timestamp": timestamp})
                last_saved_frame = frame
                last_saved_frame_count = frame_count
                saved_count += 1

                if saved_count >= max_keyframes:
                    print(f"Hit the safety cap of {max_keyframes} keyframes — stopping "
                          f"early (this video has a lot of on-screen change).")
                    break

        frame_count += 1

    cap.release()

    with open(f"{output_dir}/keyframes.json", "w") as f:
        json.dump(keyframes, f, indent=2)

    print(f"Extracted {saved_count} keyframes based on scene changes "
          f"(video length {duration_seconds/60:.1f} min, checked every "
          f"{check_interval_seconds}s, min gap {min_gap_seconds}s)")
    return keyframes

if __name__ == "__main__":
    extract_keyframes()