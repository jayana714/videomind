import os
import json
import base64
import cv2
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

DESCRIBE_SYSTEM_PROMPT = """You are describing a single frame grabbed from a screen-recorded
tutorial video. Describe what's visibly on screen: the application or tool being used, any
menus, panels, or tool names visible, any data shown (tables, maps, charts), and what step of
a workflow this appears to represent. Be specific and factual about what's actually visible --
don't guess at things outside the frame or assume context you can't see. Keep it to 2-4 sentences."""


def upscale_image(image_path, scale=2):
    """Upscales a frame before sending it to the vision model. This doesn't add real
    detail that wasn't captured, but giving the model more pixels to work with
    measurably helps it read small on-screen text and table values -- useful when the
    source video itself is low-resolution (here, YouTube only offered 360p for this
    video due to its current SABR streaming restriction, not a choice we made)."""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    resized = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    success, buffer = cv2.imencode(".jpg", resized)
    return buffer.tobytes()


def describe_keyframe(image_path, model="claude-haiku-4-5-20251001"):
    image_data = base64.standard_b64encode(upscale_image(image_path)).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=DESCRIBE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": "Describe this frame."},
            ],
        }],
    )
    return response.content[0].text


def describe_all_keyframes(keyframes_dir="keyframes"):
    """Adds a 'description' field to each keyframe entry, describing what's on screen --
    this is what makes keyframes searchable later, the same way transcribe.py turned
    raw audio into searchable text back in v1."""
    with open(f"{keyframes_dir}/keyframes.json") as f:
        keyframes = json.load(f)

    for kf in keyframes:
        print(f"Describing {kf['path']} ({kf['timestamp']:.1f}s)...")
        kf["description"] = describe_keyframe(kf["path"])

    with open(f"{keyframes_dir}/keyframes.json", "w") as f:
        json.dump(keyframes, f, indent=2)

    print(f"Described {len(keyframes)} keyframes.")
    return keyframes


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        describe_all_keyframes()