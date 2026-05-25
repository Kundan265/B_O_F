"""
Bodies of Flora - Stage 2: Image Generation and Cleaning

- Stage 2a: FLUX.1-schnell 2D image generation
- Stage 2b: Image cleaning (background removal, morphological cleanup)
"""

import gc
import io
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

from .config import (
    DEVICE,
    OUTPUT_DIR,
    DEFAULT_FLUX_MODEL,
    FLUX_IMAGE_SIZE,
    FLUX_FALLBACK_SIZE,
)

# Global FLUX pipeline reference
_flux_pipe = None


def _load_flux(hf_token: str = "") -> "FluxPipeline":
    """
    Load FLUX.1-schnell pipeline with CPU offloading.
    
    Uses CPU offload to keep GPU memory free for Hunyuan3D.
    
    Args:
        hf_token: Optional HuggingFace token for gated models
        
    Returns:
        Loaded FluxPipeline instance
    """
    global _flux_pipe
    
    if _flux_pipe is not None:
        return _flux_pipe

    print("  Loading FLUX.1-schnell (CPU offload)...")
    
    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()

    from diffusers import FluxPipeline
    
    token = os.environ.get("HF_TOKEN", None)
    _flux_pipe = FluxPipeline.from_pretrained(
        DEFAULT_FLUX_MODEL,
        torch_dtype=torch.float16,
        token=token
    )

    # CPU offload instead of .to(DEVICE) to save VRAM for Hunyuan3D
    _flux_pipe.enable_model_cpu_offload()
    print("  ✓ FLUX ready (CPU offload)")
    
    return _flux_pipe


def _unload_flux():
    """Unload FLUX pipeline and free GPU memory."""
    global _flux_pipe
    
    if _flux_pipe is not None:
        del _flux_pipe
        _flux_pipe = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage2a_flux(
    prompt: str,
    clip_hint: str = "",
    seed: int = 42,
    hf_token: str = ""
) -> str:
    """
    Generate 2D image with FLUX.1-schnell.
    
    Args:
        prompt: Full FLUX prompt for image generation
        clip_hint: Short CLIP-friendly prompt (optional)
        seed: Random seed for reproducibility
        hf_token: HuggingFace token for model access
        
    Returns:
        Path to generated image file
    """
    t0 = time.time()
    print("━━━ Stage 2a: FLUX.1-schnell ━━━")

    pipe = _load_flux(hf_token=hf_token)
    gen = torch.Generator("cpu").manual_seed(int(seed))

    try:
        with torch.inference_mode():
            out = pipe(
                prompt=clip_hint if clip_hint else prompt,
                prompt_2=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=gen,
                height=FLUX_IMAGE_SIZE,
                width=FLUX_IMAGE_SIZE
            )
    except RuntimeError:
        # OOM fallback to smaller size
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  OOM → retrying {FLUX_FALLBACK_SIZE}×{FLUX_FALLBACK_SIZE}")
        gen = torch.Generator("cpu").manual_seed(int(seed))
        with torch.inference_mode():
            out = pipe(
                prompt=clip_hint if clip_hint else prompt,
                prompt_2=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=gen,
                height=FLUX_FALLBACK_SIZE,
                width=FLUX_FALLBACK_SIZE
            )

    img = out.images[0].convert("RGB")
    output_path = OUTPUT_DIR / f"flux_{int(time.time())}_{seed}.png"
    img.save(output_path, "PNG")
    
    print(f"  ✓ Generated ({time.time() - t0:.1f}s)")
    return str(output_path)


def stage2b_clean(image_path: str) -> str:
    """
    4-pass image cleaning: rembg → pixel kill → morph ops → crop+center.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Path to cleaned image file
    """
    t0 = time.time()
    print("━━━ Stage 2b: Image cleaning ━━━")
    
    from rembg import remove
    import cv2

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # Pass 1: Background removal with rembg
    res = remove(arr)
    if isinstance(res, bytes):
        rgba = np.array(Image.open(io.BytesIO(res)).convert("RGBA"))
    else:
        rgba = np.array(Image.fromarray(res).convert("RGBA"))

    # Pass 2: Pixel kill (remove white/black bleed artifacts)
    rgb = rgba[:, :, :3].astype(np.float32)
    bright = rgb.mean(axis=2)
    var = rgb.var(axis=2)
    rgba[(bright > 230) & (var < 20), 3] = 0  # Nearly white
    rgba[(bright < 15) & (var < 10), 3] = 0   # Nearly black

    # Pass 3: Morphological cleanup
    alpha = rgba[:, :, 3]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha[alpha < 128] = 0
    alpha[alpha >= 128] = 255
    rgba[:, :, 3] = alpha

    # Pass 4: Crop and center on white 1024×1024 canvas
    pil = Image.fromarray(rgba)
    bbox = pil.getbbox()
    size = 1024
    
    if bbox:
        cropped = pil.crop(bbox)
        cropped.thumbnail((int(size * 0.85), int(size * 0.85)), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        canvas.paste(
            cropped,
            ((size - cropped.width) // 2, (size - cropped.height) // 2),
            cropped
        )
        out = Image.new("RGB", (size, size), (255, 255, 255))
        out.paste(canvas, mask=canvas.split()[-1])
    else:
        out = img.resize((size, size))

    output_path = OUTPUT_DIR / f"clean_{int(time.time())}.png"
    out.save(output_path, "PNG")
    
    print(f"  ✓ Cleaned ({time.time() - t0:.1f}s)")
    return str(output_path)


def unload_flux_for_hunyuan():
    """
    Free FLUX memory before running Hunyuan3D.
    
    Call this between Stage 2 and Stage 3 to avoid OOM.
    """
    print("\n▸ Freeing FLUX memory for Hunyuan3D")
    _unload_flux()
