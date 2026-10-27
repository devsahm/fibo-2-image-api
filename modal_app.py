"""
Run the Fibo 2.0 API on Modal: a GPU that exists only while it's working.
=========================================================================

main.py is used unchanged. This file just tells Modal what machine to build,
which GPU to attach, and where to keep the 30 GB model between runs.

One-time setup (on your Mac):
    pip install modal
    modal setup                               # log in, opens the browser

Download the model into Modal storage (runs on a cheap CPU machine, no GPU billed):
    modal run modal_app.py::download

Start the API:
    modal serve modal_app.py                  # dev: temporary URL, stops when you press Ctrl-C
    modal deploy modal_app.py                 # or: permanent URL, sleeps when idle

Then point the client at the URL Modal prints:
    python test_client.py --url https://YOUR-WORKSPACE--fibo-api-web.modal.run
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent
MODEL_ID = "briaai/Fibo-2.0-dev"

# The container recipe: same contents as the Dockerfile. Modal builds it once and caches it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # needed to install diffusers from the PR branch
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .env({"HF_HOME": "/models/hf"})  # Hugging Face cache lives on the volume below
    .add_local_file(HERE / "main.py", "/root/main.py")
)

# A persistent disk. The model is downloaded into it once and reused by every container.
models = modal.Volume.from_name("fibo-models", create_if_missing=True)

# HF_TOKEN, GEMINI_API_KEY and the optional API_KEY are read from your local .env and passed in as env vars.
secrets = [modal.Secret.from_dotenv(HERE)]

app = modal.App("fibo-api", image=image)


@app.function(volumes={"/models": models}, secrets=secrets, timeout=60 * 60)
def download():
    """Pull the ~30 GB model into the volume. CPU only, so you don't pay GPU rates to wait on the network."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    models.commit()
    print("Model saved to the fibo-models volume.")


@app.function(
    gpu="L40S",  # 48 GB VRAM. "A100-80GB" or "H100" also work (set FLASH_ATTENTION_3=1 for H100)
    memory=32 * 1024,  # system RAM in MB: weights pass through RAM on their way to the GPU
    volumes={"/models": models},
    secrets=secrets,
    max_containers=1,  # one GPU, like --workers 1: never load two copies of the model
    scaledown_window=5 * 60,  # shut the GPU off after 5 idle minutes
    startup_timeout=20 * 60,  # loading the model can take several minutes
    timeout=15 * 60,  # longest a single request may run
)
@modal.concurrent(max_inputs=8)  # let requests share the container; main.py's semaphore queues them for the GPU
@modal.asgi_app()
def web():
    from main import app as fastapi_app

    return fastapi_app
