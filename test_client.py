"""
Quick test: generate an image, then edit it.

    pip install requests
    python test_client.py --url http://YOUR_SERVER_IP:8000 --key YOUR_API_KEY
"""
import argparse
import base64
import json

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--key", default="")
args = p.parse_args()
headers = {"X-API-Key": args.key} if args.key else {}

print("Health:", requests.get(f"{args.url}/health", timeout=30).json())

# 1) Text-to-image
print("Generating... (about 20-60 s depending on GPU)")
r = requests.post(
    f"{args.url}/v1/text-to-image",
    headers=headers,
    json={
        "prompt": "A Lagos street market at golden hour, colourful umbrellas, vendors selling fruit, photorealistic",
        "width": 1424,
        "height": 800,
        "seed": 42,
    },
    timeout=600,
)
r.raise_for_status()
data = r.json()
with open("generated.png", "wb") as f:
    f.write(base64.b64decode(data["image_base64"]))
with open("generated_blueprint.json", "w") as f:
    json.dump(data["structured_prompt"], f, indent=2)
print(f"Saved generated.png (seed {data['seed']}, {data['inference_seconds']} s)")

# 2) Image-to-image (edit the image we just made)
print("Editing...")
with open("generated.png", "rb") as img:
    r = requests.post(
        f"{args.url}/v1/image-editing",
        headers=headers,
        files=[("images", ("generated.png", img, "image/png"))],
        data={"prompt": "make it night time with warm street lights", "seed": "42"},
        timeout=600,
    )
r.raise_for_status()
data = r.json()
with open("edited.png", "wb") as f:
    f.write(base64.b64decode(data["image_base64"]))
print(f"Saved edited.png ({data['inference_seconds']} s). Edit understood as: {data['edit_instruction']}")
