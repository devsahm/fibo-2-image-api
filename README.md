# Fibo 2.0 Image API

A small web API that generates and edits images with [Bria's Fibo 2.0](https://huggingface.co/briaai/Fibo-2.0-dev) model.

- **Text-to-image:** send a sentence, get an image.
- **Image editing:** send a photo plus an instruction ("make the car red"), get the edited photo.

> ⚠️ The model is licensed for **non-commercial use only** (CC BY-NC 4.0).

## How it works

```text
your prompt ──► Gemini writes a detailed JSON "blueprint" ──► Fibo 2.0 paints it on a GPU ──► PNG
```

Fibo 2.0 was trained on long JSON descriptions (objects, lighting, camera, style), not short prompts. The API uses Google Gemini to turn your sentence into one. Each response includes that blueprint. Edit a field, send it back as `structured_prompt`, and only that part of the image changes.

The model needs a GPU with about **30 GB of memory**. A normal laptop (including Macs) can't run it. This guide uses [Modal](https://modal.com), which rents you a GPU by the second and switches it off when idle.

---

## 1. Get your keys

| Key | Where to get it |
| --- | --- |
| `HF_TOKEN` | 1. Open the [model page](https://huggingface.co/briaai/Fibo-2.0-dev), log in, and submit the access form. 2. Wait until the page says you have access. 3. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

Copy the example config and paste both keys in:

```bash
cp .env.example .env
```

Never commit `.env`. It's already in `.gitignore`.

## 2. Run it on Modal

```bash
pip install modal
modal setup                          # one-time: log in to Modal in your browser
modal run modal_app.py::download     # one-time: download the model (~30 GB, 10–20 min)
modal serve modal_app.py             # start the API
```

`modal serve` prints a URL like `https://your-name--fibo-api-web-dev.modal.run`. Keep the terminal open. Press **Ctrl-C** to stop.

Open `<URL>/health` in your browser. The first time, it takes 1–3 minutes to load the model. Wait until you see `"loaded": true`.

## 3. Try it

**In the browser (easiest):** open `<URL>/docs`, pick an endpoint, click **Try it out**, then **Execute**.

**With the test script** (generates an image, then edits it):

```bash
pip install requests
python test_client.py --url <URL>
```

**With curl:**

```bash
# Text-to-image
curl -X POST "<URL>/v1/text-to-image?format=png" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "An origami whale swimming through paper clouds", "seed": 7}' \
  -o whale.png

# Image editing
curl -X POST "<URL>/v1/image-editing?format=png" \
  -F "images=@photo.jpg" \
  -F "prompt=replace the jacket with a red leather one" \
  -o edited.png
```

## Endpoints

| Endpoint | Send | Get back |
| --- | --- | --- |
| `POST /v1/text-to-image` | JSON: `prompt` | New image |
| `POST /v1/image-editing` | Form: 1 image + `prompt` | Edited image |
| | 1 image + `mask` + `prompt` | Only the white area of the mask changes |
| | 2–4 images + `prompt` (refer to them as `<image_1>`, `<image_2>`) | Combined image, e.g. virtual try-on |
| `GET /health` | | Whether the model is loaded, GPU memory use |

Add `?format=png` to get a raw PNG. Without it you get JSON with the image as base64, plus the seed and blueprint.

**Optional settings:**

| Setting | Default | What it does |
| --- | --- | --- |
| `seed` | random | Same seed + same settings = same image |
| `steps` | 50 | More steps means more detail, but slower |
| `guidance_scale` | 5.0 | How strictly to follow the prompt |
| `width`, `height` | 1024 | Keep near 1 megapixel: `1024×1024`, `1424×800`, `1248×832` (text-to-image only) |

---

## Running on your own GPU server instead

On any Linux machine with an NVIDIA GPU (48 GB recommended, e.g. L40S or A100), for example rented from RunPod or Vast.ai:

```bash
docker build -t fibo-api .
docker run --gpus all -p 8000:8000 --env-file .env -v fibo-models:/models fibo-api
```

The API is then at `http://SERVER_IP:8000`. On a 24 GB GPU, set `CPU_OFFLOAD=1` in `.env` (slower).

## Costs

- **Modal:** about $2/hour for the GPU, billed only while running. Each start costs 1–3 minutes of model loading, then 15–60 seconds per image. New accounts get free monthly credit.
- **Gemini:** a small charge per plain-text prompt. Requests that send `structured_prompt` skip Gemini.
- Your Modal URL is public, and anyone who has it can use your credit. Don't share it.

## Troubleshooting

| Error | Fix |
| --- | --- |
| `not in the authorized list` / `403` | Your Hugging Face access hasn't been approved yet. Check the model page. |
| `502 Prompt writer failed` | Check `GEMINI_API_KEY` and its quota. |
| `test_client.py` times out | The model is still loading. Wait for `/health` to show `"loaded": true`. |
| `GPU ran out of memory` | Use a smaller image size, or set `CPU_OFFLOAD=1`. |
| `ImportError: BriaFibo2Pipeline` | Reinstall dependencies: `pip install -r requirements.txt` |
