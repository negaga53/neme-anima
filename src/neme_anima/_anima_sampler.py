"""Standalone Anima + LoRA sample generator.

Runs inside **diffusion-pipe's venv**, never neme-anima's: it imports the
ComfyUI copy vendored at ``<diffusion_pipe_dir>/submodules/ComfyUI`` and uses
it purely as a library (no ComfyUI server). That copy supports Anima (DiT,
Qwen3-0.6B text encoder, Qwen-Image VAE) and exposes the exact KSampler
sampler / scheduler names the UI offers.

Invoked by ``neme_anima.server.sampling_runner`` as::

    <venv python> _anima_sampler.py --job <job.json>

The job file (see ``neme_anima.training.build_sample_job``) carries the model
paths, the LoRA path, the prompts, sampler settings and the output directory.
The script writes ``p00.png, p01.png, ...`` (one per prompt) plus a
``manifest.json`` into ``output_dir``. The manifest is written *last* — its
presence is how the runner knows an epoch has been sampled. On failure a
manifest with an ``error`` field is still written, then the script exits 1.

ComfyUI's memory manager decides how much of each model fits in the VRAM that
is free *right now* (training may be holding most of it) and offloads the
rest, so the same script works concurrently with a run or after it.

This file must not import anything from ``neme_anima``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _setup_comfy(diffusion_pipe_dir: str) -> None:
    comfy_root = Path(diffusion_pipe_dir).expanduser() / "submodules" / "ComfyUI"
    if not (comfy_root / "comfy").is_dir():
        raise RuntimeError(f"vendored ComfyUI not found at {comfy_root}")
    sys.path.insert(0, str(comfy_root))
    # ComfyUI parses its CLI flags at import time; don't let it see ours.
    sys.argv = [sys.argv[0]]
    # diffusion-pipe's venv doesn't ship torchsde. ComfyUI imports it at
    # module level but only the SDE samplers (dpmpp_sde, dpmpp_2m_sde, ...)
    # touch it, so stub it with a module that fails loudly on actual use.
    try:
        import torchsde  # noqa: F401
    except ImportError:
        import types

        stub = types.ModuleType("torchsde")

        def _missing(*_a, **_k):
            raise RuntimeError(
                "this sampler needs torchsde, which is not installed in the "
                "diffusion-pipe venv (pip install torchsde there, or pick a "
                "non-SDE sampler)",
            )

        stub.BrownianInterval = _missing
        stub.BrownianTree = _missing
        sys.modules["torchsde"] = stub


def _run(job: dict, out_dir: Path) -> list[dict]:
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import numpy as np
    import torch
    from PIL import Image

    s = job["settings"]
    if s["sampler"] not in comfy.samplers.KSampler.SAMPLERS:
        raise ValueError(f"unknown sampler {s['sampler']!r}")
    if s["scheduler"] not in comfy.samplers.KSampler.SCHEDULERS:
        raise ValueError(f"unknown scheduler {s['scheduler']!r}")

    t0 = time.time()
    model = comfy.sd.load_diffusion_model(job["dit_path"])
    clip = comfy.sd.load_clip([job["llm_path"]])
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(job["vae_path"]))
    lora = comfy.utils.load_torch_file(job["lora_path"], safe_load=True)
    model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0.0)
    print(f"[sampler] models loaded in {time.time() - t0:.1f}s", flush=True)

    def encode(text: str):
        return clip.encode_from_tokens_scheduled(clip.tokenize(text))

    negative = encode(job.get("negative_prompt", ""))
    width, height = int(s["width"]), int(s["height"])
    images: list[dict] = []
    for i, prompt in enumerate(job["prompts"]):
        t1 = time.time()
        positive = encode(prompt)
        latent = torch.zeros(
            [1, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        noise = comfy.sample.prepare_noise(latent, int(s["seed"]))
        samples = comfy.sample.sample(
            model, noise, int(s["steps"]), float(s["cfg"]),
            s["sampler"], s["scheduler"], positive, negative, latent,
            denoise=1.0, seed=int(s["seed"]), disable_pbar=True,
        )
        decoded = vae.decode(samples)
        # Video-shaped VAEs return [B, T, H, W, C]; flatten to frames.
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, *decoded.shape[-3:])
        arr = (decoded[0].clamp(0, 1).cpu().float().numpy() * 255.0).round()
        name = f"p{i:02d}.png"
        Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)
        elapsed = time.time() - t1
        print(f"[sampler] prompt {i + 1}/{len(job['prompts'])} done in {elapsed:.1f}s",
              flush=True)
        images.append({"file": name, "prompt": prompt, "seconds": round(elapsed, 2)})
    return images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text())
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "epoch": job.get("epoch"),
        "checkpoint": job.get("checkpoint"),
        "prompts": job["prompts"],
        "negative_prompt": job.get("negative_prompt", ""),
        "settings": job["settings"],
        "images": [],
        "error": None,
        "started_at": time.time(),
    }
    rc = 0
    try:
        _setup_comfy(job["diffusion_pipe_dir"])
        import torch

        # ComfyUI's server runs every node under inference_mode; without it
        # the VAE output carries autograd state and wastes VRAM.
        with torch.inference_mode():
            manifest["images"] = _run(job, out_dir)
    except Exception as e:  # noqa: BLE001 — every failure must land in the manifest
        traceback.print_exc()
        manifest["error"] = f"{type(e).__name__}: {e}"
        rc = 1
    manifest["finished_at"] = time.time()
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out_dir / "manifest.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
