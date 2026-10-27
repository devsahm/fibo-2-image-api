# Fibo 2.0 Image API (FastAPI + GPU)

Text-to-image and image-editing API for [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev).

> ⚠️ **License: non-commercial only (CC BY-NC 4.0).** You can build, test and demo freely.
> To charge users or run it inside a paid product, you need a commercial licence from Bria:
> https://platform.bria.ai/licensing/pricing?model=fibo-2

---

## How it works (plain language)

Think of it as a small print shop with three workers.

**1. The receptionist (FastAPI).** Your app sends a request: "make me a picture of X" or "here's my photo, change Y". The receptionist checks the request makes sense and hands it on.

**2. The planner (bria-vgl + Gemini).** Fibo 2.0 is unusual. It wasn't trained on short prompts like "a cat on a sofa". It was trained on very detailed *blueprints*: JSON documents describing every object, the lighting, camera angle, colours, style, and so on. The planner takes your short sentence and writes that detailed blueprint for you. It uses Google's Gemini to do this, which is why you need a Gemini key.
- You get the blueprint back with every image. Change one field (say, "lighting") and re-send it as `structured_prompt` to change *only* that, without the rest of the picture drifting. That's Fibo's superpower.
- If you send your own blueprint, the planner is skipped and no Gemini call happens.

**3. The painter (the model on the GPU).** The painter reads the blueprint and paints the image. It starts from pure random noise and cleans it up a little at a time. Each clean-up pass is a **step**. 50 steps is the recommended quality. Painting needs a powerful graphics card (GPU), because each step is billions of calculations.

**The painter is loaded only once.** When the server starts, the ~30 GB model is copied into the GPU's memory. That takes a few minutes, but after that every request reuses it.

**There's a queue.** The GPU paints one picture at a time (configurable). If 10 people ask at once, they wait in line instead of crashing the server.

### The two endpoints

| Endpoint | You send | You get |
|---|---|---|
| `POST /v1/text-to-image` | A sentence | A new image + its blueprint |
| `POST /v1/image-editing` | 1 image + instruction ("make the car red") | The edited image. Everything you didn't mention stays the same |
| | 1 image + **mask** + instruction | Only the white area of the mask changes |
| | 2–4 images + instruction ("put the person in `<image_1>` in the dress from `<image_2>`") | A combined image (virtual try-on, product placement, style transfer) |

### Settings you can tweak

- **steps**: more steps means more detail and slower generation. Default 50.
- **guidance_scale**: how strictly to follow the prompt. Default 5.0.
- **seed**: a number for the randomness. Same seed + same settings = the same image again.
- **width / height**: keep the total around 1 megapixel: `1024×1024` (square), `1424×800` (16:9), `1248×832` (3:2).

---

## What GPU do I need?

The model has three parts that all sit in GPU memory: the 8B painter (~16 GB), the Qwen3-VL-4B text reader (~8–9 GB), and the image decoder. Plan for roughly **30 GB+ of VRAM**. *(Estimate. Check `/health` for real usage once it's running.)*

| GPU | VRAM | Setting | Notes |
|---|---|---|---|
| L40S / RTX A6000 | 48 GB | default | Good value choice |
| A100 | 40/80 GB | default | Solid |
| H100 | 80 GB | `FLASH_ATTENTION_3=1` | Fastest |
| RTX 4090 / L4 / A10 | 24 GB | `CPU_OFFLOAD=1` | Works but slower; needs 64 GB+ system RAM |

Where to rent: RunPod, Vast.ai, Lambda, or AWS/GCP. Pick a template with Docker + NVIDIA drivers, or a PyTorch template.

There are also community quantized (smaller) versions of the model on its Hugging Face page. They're useful if you want to fit a 24 GB card without offload, at some quality cost.

---

## Setup, step by step

### Step 1: Get access to the model
1. Create a Hugging Face account and log in.
2. Open https://huggingface.co/briaai/Fibo-2.0-dev and fill in the access form.
3. Wait until the page says you have access. Until then, downloads fail with *"you are not in the authorized list"*.
4. Create a token at https://huggingface.co/settings/tokens (Read access).

### Step 2: Get a Gemini key
https://aistudio.google.com/apikey. The planner uses it for every request that sends a plain-language prompt.

### Step 3: Rent a GPU server and copy this folder to it
```bash
scp -r fibo-api/ root@YOUR_SERVER_IP:~/
ssh root@YOUR_SERVER_IP
cd fibo-api
cp .env.example .env
nano .env        # paste HF_TOKEN and GEMINI_API_KEY
```

### Step 4a: Run with Docker (recommended)
```bash
docker build -t fibo-api .
docker run --gpus all -p 8000:8000 --env-file .env \
  -v fibo-models:/models fibo-api
```
The `-v fibo-models:/models` part saves the downloaded model, so restarts don't re-download 30 GB.

### Step 4b: Or run without Docker
```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
set -a && source .env && set +a
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait for the log line `Model ready on ...`. The first start downloads the model, so it can take 5–15 minutes.

### Step 5: Try it
- Interactive docs (test right in your browser): `http://YOUR_SERVER_IP:8000/docs`
- Or run `python test_client.py --url http://YOUR_SERVER_IP:8000`

---

## Option B: Run on Modal (no GPU rental)

[Modal](https://modal.com) starts a GPU only while a request is running and turns it off when idle. You pay per second, and new accounts get free monthly credit, so a short test costs little or nothing. `modal_app.py` runs the same `main.py`. Nothing else changes.

Do Steps 1 and 2 above first, and fill in `.env` on your own computer. Then:

```bash
pip install modal
modal setup                          # one-time login in the browser
modal run modal_app.py::download     # copy the model into Modal storage (CPU only, no GPU billed)
modal serve modal_app.py             # prints a URL like https://YOUR-WORKSPACE--fibo-api-web-dev.modal.run
```

1. Open `<URL>/health` in your browser. The first time, it takes 1–3 minutes while the model loads onto the GPU. Wait until it shows `"loaded": true`.
2. Then test with `<URL>/docs`, `python test_client.py --url <URL>`, or the curl examples below with `<URL>` in place of `http://localhost:8000`.
3. Press Ctrl-C to stop. The GPU also switches itself off after 5 idle minutes.

Good to know:
- `modal serve` exists only while your terminal is open. `modal deploy modal_app.py` gives a permanent URL that still costs nothing while idle.
- The URL is public: anyone who has it can use your GPU credit, so don't share it.
- Your `.env` values (HF and Gemini keys) are sent to Modal as environment variables.
- Every cold start (first request after idle) reloads the model, and you pay for that time.

---

## API examples

**Text-to-image (JSON response with base64 image)**
```bash
curl -X POST http://localhost:8000/v1/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "An origami whale swimming through paper clouds", "width": 1424, "height": 800, "seed": 7}'
```

**Text-to-image (straight to a PNG file)**
```bash
curl -X POST "http://localhost:8000/v1/text-to-image?format=png" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A modern Nigerian living room, afternoon light"}' -o out.png
```

**Edit an image**
```bash
curl -X POST "http://localhost:8000/v1/image-editing?format=png" \
  -F "images=@photo.jpg" -F "prompt=replace the jacket with a red leather one" -o edited.png
```

**Masked edit** (white = change, black = keep)
```bash
curl -X POST "http://localhost:8000/v1/image-editing?format=png" \
  -F "images=@room.jpg" -F "mask=@mask.png" -F "prompt=replace the sofa with a green velvet one" -o edited.png
```

**Multi-reference (up to 4 images)**
```bash
curl -X POST "http://localhost:8000/v1/image-editing?format=png" \
  -F "images=@person.png" -F "images=@dress.png" \
  -F "prompt=dress the person in <image_1> in the dress from <image_2>" -o tryon.png
```

**Re-render a tweaked blueprint** (no Gemini call)
```bash
# take structured_prompt from a previous JSON response, edit e.g. the "lighting" field, send it back:
curl -X POST http://localhost:8000/v1/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"structured_prompt": { ... }, "seed": 7}'
```

**JSON response shape**
```json
{
  "image_base64": "iVBORw0KGgo...",
  "mime_type": "image/png",
  "seed": 7,
  "width": 1424, "height": 800,
  "steps": 50, "guidance_scale": 5.0,
  "inference_seconds": 31.4,
  "structured_prompt": { "short_description": "...", "objects": [ ... ], "lighting": { ... } },
  "edit_instruction": null
}
```

---

## Good to know

- **Speed:** with 50 steps, expect roughly 15–60 s per image depending on the GPU. For faster *editing*, Bria offers [Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) (few steps, no guidance). Swap `MODEL_ID` and lower the steps.
- **Gemini is called on every plain prompt.** That adds a second or two and a small cost, and the prompt and uploaded images are sent to Google. Send `structured_prompt` to avoid it.
- **One worker only.** Never run uvicorn with `--workers 2+`, because each worker loads its own 30 GB copy. To scale, run more GPU servers behind a load balancer.
- **Uploaded images are resized to ~1 megapixel**, the size the model was trained at.
- **diffusers comes from a PR branch** until Bria's pipeline is merged into an official release. When it is, pin the release version in `requirements.txt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `401 Repository not found / gated` | Accept the model terms on Hugging Face; make sure `HF_TOKEN` is set |
| `No CUDA GPU detected` | Docker needs `--gpus all`; check `nvidia-smi` works |
| `GPU ran out of memory` | Set `CPU_OFFLOAD=1`, use a smaller size, or a bigger GPU |
| `502 Prompt writer failed` | Check `GEMINI_API_KEY` and its quota |
| `you are not in the authorized list` | Your Hugging Face account hasn't been approved for the model yet. Request access on the model page and wait |
| `test_client.py` times out on Modal | The GPU is cold-starting. Open `/health` in a browser first and wait for `"loaded": true` |
| `ImportError: BriaFibo2Pipeline` | diffusers wasn't installed from the PR branch. Re-run `pip install -r requirements.txt` |
