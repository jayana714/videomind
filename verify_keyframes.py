import os
import json
import base64
import cv2
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

VERIFY_SYSTEM_PROMPT = """You are fact-checking a description of an image against the actual image.
Look carefully at the image and the description below. Check whether every specific detail in the
description -- place names, tool names, labels, text, numbers -- is actually visible in the image.
Be especially alert to invented specifics: a made-up place name, a label that isn't actually shown,
or a detail confused with something else in the image (for example, mistaking a proper noun that's
part of a label -- like a school name -- for an actual place name).

Respond in exactly this format:
Grounded: yes or no
Explanation: one or two sentences. If "no", name exactly which claim is not actually visible."""


def upscale_image(image_path, scale=2):
    """Same upscaling as describe_keyframes.py -- verification needs to see the same
    image quality that generation used, or it would be checking against a worse view
    than the one the description was actually written from."""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    resized = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    success, buffer = cv2.imencode(".jpg", resized)
    return buffer.tobytes()


def verify_keyframe_description(image_path, description, model="claude-haiku-4-5-20251001"):
    image_data = base64.standard_b64encode(upscale_image(image_path)).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=VERIFY_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"Description to check:\n{description}"},
            ],
        }],
    )
    return response.content[0].text


CORRECTION_SYSTEM_PROMPT = """You wrote a description of an image, and a fact-checker found a
specific problem with it. Look at the image again and fix ONLY the issue described -- keep
everything else in the description that was already correct. Output the complete corrected
description, not just the fixed part."""


def correct_keyframe_description(image_path, description, explanation, model="claude-haiku-4-5-20251001"):
    image_data = base64.standard_b64encode(upscale_image(image_path)).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=CORRECTION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"Original description:\n{description}\n\nProblem found:\n{explanation}\n\nCorrected description:"},
            ],
        }],
    )
    return response.content[0].text


def verify_all_keyframes(keyframes_dir="keyframes"):
    """Re-checks each description against its actual image, and auto-corrects (one
    attempt, same cap as the summary correction pass) anything that doesn't hold up --
    the visual equivalent of Step 9's transcript verification."""
    with open(f"{keyframes_dir}/keyframes.json") as f:
        keyframes = json.load(f)

    for kf in keyframes:
        print(f"Verifying {kf['path']}...")
        verification = verify_keyframe_description(kf["path"], kf["description"])
        grounded = "yes" in verification.lower().split("\n")[0]

        if not grounded:
            explanation = verification.split("Explanation:", 1)[-1].strip()
            print(f"  ⚠️  Issue found: {explanation}")
            print(f"  Correcting...")
            kf["description"] = correct_keyframe_description(kf["path"], kf["description"], explanation)
        else:
            print(f"  ✅ Verified")

    with open(f"{keyframes_dir}/keyframes.json", "w") as f:
        json.dump(keyframes, f, indent=2)

    print(f"Done verifying {len(keyframes)} keyframe descriptions.")
    return keyframes


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your .env file first.")
    else:
        verify_all_keyframes()