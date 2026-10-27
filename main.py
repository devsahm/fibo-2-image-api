"""
Fibo 2.0 Dev — Text-to-Image & Image-to-Image API
=================================================

A FastAPI server that loads briaai/Fibo-2.0-dev onto a GPU once at startup and
exposes two endpoints:

  POST /v1/text-to-image   -> JSON body, returns a generated image
  POST /v1/image-editing   -> JSON body (1-4 images as base64 or public URLs, optional mask), returns an edited image

How a request flows:
  1. Your plain-language prompt goes to bria-vgl, which (via Gemini) rewrites it
     into the long structured JSON "blueprint" (VGL) that Fibo 2.0 was trained on.
     You can skip this step by sending your own `structured_prompt`.
  2. The blueprint is handed to the GPU pipeline, which "paints" the image.
  3. The image comes back as base64 JSON (default) or raw PNG (?format=png).

Run with ONE worker only — every worker would load its own ~30 GB copy of the model:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import random
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------------------
# Configuration (all via environment variables — see .env.example)
# --------------------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "briaai/Fibo-2.0-dev")
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"  # set to 1 on 24 GB cards (slower)
FLASH_ATTENTION_3 = os.getenv("FLASH_ATTENTION_3", "0") == "1"  # H100 only
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "50"))  # model card recommends 50
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "5.0"))  # model card recommends 5.0
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
TARGET_PIXELS = 1024 * 1024  # Fibo 2.0 was trained at ~1 megapixel

# Negative prompts from the official model card.
# Text-to-image: steer away from soft, plastic, artifact-laden images.
NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified '
    'shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, '
    'no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast",'
    '"overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":'
    '{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, '
    'lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'
)
# Edits: the empty caption the model was trained with.
NEGATIVE_EDIT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},'
    '"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}'
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fibo-api")

# Shared state filled at startup
STATE: dict[str, Any] = {}
# The "queue": only MAX_CONCURRENT_JOBS images are painted on the GPU at the same time.
GPU_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def load_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Fibo 2.0 needs an NVIDIA GPU.")

    from diffusers import BriaFibo2Pipeline  # available from the diffusers PR branch (see requirements.txt)
    from huggingface_hub import hf_hub_download

    log.info("Loading %s (this takes a few minutes the first time — it downloads ~30 GB)...", MODEL_ID)
    pipe = BriaFibo2Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

    if CPU_OFFLOAD:
        log.info("CPU offload enabled: parts of the model wait in RAM and move to the GPU when needed.")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if FLASH_ATTENTION_3:
        log.info("Using FlashAttention-3 backend (H100).")
        pipe.transformer.set_attention_backend("_flash_3_varlen_hub")

    # Fibo 2.0 "sees" any text in the caption drawn as a small image, so it needs these fonts.
    fonts = [
        hf_hub_download(MODEL_ID, name, subfolder="fonts")
        for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")
    ]
    log.info("Model ready on %s.", torch.cuda.get_device_name(0))
    return pipe, fonts


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe, fonts = await run_in_threadpool(load_pipeline)
    STATE["pipe"], STATE["fonts"] = pipe, fonts
    yield
    STATE.clear()
    torch.cuda.empty_cache()


app = FastAPI(
    title="Fibo 2.0 Image API",
    version="1.0.0",
    description="Text-to-image and image editing on briaai/Fibo-2.0-dev.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def snap16(x: int) -> int:
    """Round to the nearest multiple of 16 (the model works in 16-pixel blocks)."""
    return max(16, int(round(x / 16)) * 16)


def fit_to_megapixel(img: Image.Image, resample=Image.LANCZOS) -> Image.Image:
    """Resize to ~1 MP keeping aspect ratio, with sides that are multiples of 16."""
    w, h = img.size
    scale = math.sqrt(TARGET_PIXELS / (w * h))
    new_w, new_h = snap16(w * scale), snap16(h * scale)
    return img if (new_w, new_h) == (w, h) else img.resize((new_w, new_h), resample)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fetch_url(url: str, limit: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fibo-api"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(limit + 1)


async def load_image(src: str, name: str, mode: str) -> Image.Image:
    """Public http(s) URL or base64 string (a "data:image/...;base64," prefix is allowed) -> PIL image."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    if src.startswith(("http://", "https://")):
        try:
            data = await run_in_threadpool(fetch_url, src, limit)
        except Exception as e:
            raise HTTPException(400, f"{name}: could not download {src} ({e}).")
    else:
        try:
            data = base64.b64decode(src.split(",", 1)[-1], validate=True)
        except ValueError:
            raise HTTPException(400, f"{name}: not a valid URL or base64 string.")
    if len(data) > limit:
        raise HTTPException(413, f"{name}: image larger than {MAX_UPLOAD_MB} MB.")
    try:
        return Image.open(io.BytesIO(data)).convert(mode)
    except UnidentifiedImageError:
        raise HTTPException(400, f"{name}: not a readable image.")


def scene_to_json(scene) -> str:
    """bria-vgl's scene object -> the JSON string the pipeline reads."""
    return json.dumps(scene.model_dump(exclude_none=True), ensure_ascii=False)


def normalize_structured_prompt(sp: Union[str, dict]) -> str:
    if isinstance(sp, dict):
        return json.dumps(sp, ensure_ascii=False)
    try:
        json.loads(sp)
    except json.JSONDecodeError:
        raise HTTPException(400, "structured_prompt must be valid JSON.")
    return sp


async def run_vgl(fn, payload, seed: int):
    """Call bria-vgl (Gemini) off the event loop. This step does NOT use the GPU."""
    try:
        return await run_in_threadpool(fn, payload, seed=seed)
    except Exception as e:  # network / Gemini / quota errors
        log.exception("bria-vgl failed")
        raise HTTPException(502, f"Prompt writer (bria-vgl/Gemini) failed: {e}")


def _render(**kwargs) -> Image.Image:
    with torch.inference_mode():
        return STATE["pipe"](fonts=STATE["fonts"], **kwargs).images[0]


async def run_on_gpu(**kwargs) -> tuple[Image.Image, float]:
    """Wait for a free GPU slot, then paint the image in a background thread."""
    if "pipe" not in STATE:
        raise HTTPException(503, "Model is still loading. Try again shortly.")
    async with GPU_SLOTS:
        t0 = time.perf_counter()
        try:
            image = await run_in_threadpool(_render, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU ran out of memory. Use a smaller size, or set CPU_OFFLOAD=1.")
        return image, round(time.perf_counter() - t0, 2)


class ImageResult(BaseModel):
    image_base64: str = Field(description="PNG image, base64-encoded")
    mime_type: str = "image/png"
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    inference_seconds: float
    structured_prompt: dict = Field(description="The full VGL blueprint used — reuse it to re-render or tweak")
    edit_instruction: Optional[str] = None


def build_response(img, caption, seed, steps, guidance, seconds, fmt, edit_instruction=None):
    png = image_to_png_bytes(img)
    if fmt == "png":
        return Response(
            content=png,
            media_type="image/png",
            headers={"X-Seed": str(seed), "X-Inference-Seconds": str(seconds)},
        )
    return ImageResult(
        image_base64=base64.b64encode(png).decode(),
        seed=seed,
        width=img.width,
        height=img.height,
        steps=steps,
        guidance_scale=guidance,
        inference_seconds=seconds,
        structured_prompt=json.loads(caption),
        edit_instruction=edit_instruction,
    )


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/health")
async def health():
    info: dict[str, Any] = {"model": MODEL_ID, "loaded": "pipe" in STATE, "cpu_offload": CPU_OFFLOAD}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(
            gpu=torch.cuda.get_device_name(0),
            vram_used_gb=round((total - free) / 1e9, 1),
            vram_total_gb=round(total / 1e9, 1),
        )
    return info


class TextToImageRequest(BaseModel):
    prompt: Optional[str] = Field(
        None, description="Plain-language idea. bria-vgl expands it into a structured blueprint (needs GEMINI_API_KEY)."
    )
    structured_prompt: Optional[Union[dict, str]] = Field(
        None, description="Advanced: your own VGL JSON. Skips the Gemini step entirely."
    )
    width: int = Field(1024, ge=256, le=2048, description="Keep width*height near 1 MP, e.g. 1024x1024, 1424x800, 1248x832")
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=100, description="More steps = better detail, slower")
    guidance_scale: float = Field(DEFAULT_GUIDANCE, ge=0.0, le=20.0, description="How strictly to follow the prompt")
    seed: Optional[int] = Field(None, ge=0, description="Same seed + same settings = same image")

    @model_validator(mode="after")
    def _need_a_prompt(self):
        if not self.prompt and not self.structured_prompt:
            raise ValueError("Provide either 'prompt' or 'structured_prompt'.")
        return self


@app.post("/v1/text-to-image", response_model=ImageResult)
async def text_to_image(
    req: TextToImageRequest,
    format: Literal["json", "png"] = Query("json", description="json = base64 + metadata, png = raw image"),
):
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    # Step 1 — write the blueprint
    if req.structured_prompt:
        caption = normalize_structured_prompt(req.structured_prompt)
    else:
        from bria_vgl import create
        from bria_vgl.dialects.thales import Generate

        scene = await run_vgl(create, Generate(prompt=req.prompt), seed)
        caption = scene_to_json(scene)

    # Step 2 — paint it on the GPU
    image, seconds = await run_on_gpu(
        prompt=caption,
        negative_prompt=NEGATIVE_T2I,
        width=snap16(req.width),
        height=snap16(req.height),
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return build_response(image, caption, seed, req.steps, req.guidance_scale, seconds, format)


class ImageEditingRequest(BaseModel):
    images: List[str] = Field(
        ..., min_length=1, max_length=4,
        description="1 source image, or up to 4 references (<image_1>, <image_2>...), as public URLs or base64 strings",
    )
    prompt: Optional[str] = Field(None, description="Edit instruction, e.g. 'replace the jacket with a red leather one'")
    mask: Optional[str] = Field(None, description="Optional mask (URL or base64): white = area to change, black = keep. Single image only.")
    structured_prompt: Optional[Union[dict, str]] = Field(None, description="Advanced: your own VGL JSON. Skips Gemini.")
    steps: int = Field(DEFAULT_STEPS, ge=1, le=100)
    guidance_scale: float = Field(DEFAULT_GUIDANCE, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if not self.prompt and not self.structured_prompt:
            raise ValueError("Provide either 'prompt' or 'structured_prompt'.")
        if self.mask and len(self.images) != 1:
            raise ValueError("A mask can only be used with exactly one source image.")
        return self


@app.post("/v1/image-editing", response_model=ImageResult)
async def image_editing(
    req: ImageEditingRequest,
    format: Literal["json", "png"] = Query("json", description="json = base64 + metadata, png = raw image"),
):
    prompt, structured_prompt = req.prompt, req.structured_prompt
    steps, guidance_scale = req.steps, req.guidance_scale
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    # Decode and resize inputs to ~1 MP (the size the model was trained at)
    sources = [fit_to_megapixel(await load_image(src, f"images[{i}]", "RGB")) for i, src in enumerate(req.images)]
    mask_img = None
    if req.mask:
        mask_img = (await load_image(req.mask, "mask", "L")).resize(sources[0].size, Image.NEAREST)

    # Step 1 — write the edit blueprint
    edit_instruction = None
    if structured_prompt:
        caption = normalize_structured_prompt(structured_prompt)
    else:
        from bria_vgl import modify
        from bria_vgl.dialects.thales import EditImages, EditImageWithMask

        if mask_img is not None:
            payload = EditImageWithMask(prompt=prompt, image=sources[0], mask=mask_img)
        else:
            payload = EditImages(prompt=prompt, images=sources)
        scene = await run_vgl(modify, payload, seed)
        caption = scene_to_json(scene)
        edit_instruction = getattr(scene, "edit_instruction", None)

    # Step 2 — paint the edit on the GPU
    kwargs: dict[str, Any] = dict(
        prompt=caption,
        negative_prompt=NEGATIVE_EDIT,
        image=sources[0] if len(sources) == 1 else sources,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    if mask_img is not None:
        kwargs["mask"] = mask_img

    image, seconds = await run_on_gpu(**kwargs)
    return build_response(image, caption, seed, steps, guidance_scale, seconds, format, edit_instruction)


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
