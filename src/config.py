"""
Bodies of Flora - Configuration

Centralized configuration for paths, device settings, and constants.
All paths can be overridden via environment variables.
"""

import os
import sys
from pathlib import Path

import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths - configurable via environment variables
HUNYUAN3D_PATH = Path(os.environ.get("HUNYUAN3D_PATH", "./Hunyuan3D-2"))
OUTPUT_DIR = Path(os.environ.get("BOF_OUTPUT_DIR", "./outputs"))

# Ensure Hunyuan3D is in path if it exists
if HUNYUAN3D_PATH.exists() and str(HUNYUAN3D_PATH) not in sys.path:
    sys.path.insert(0, str(HUNYUAN3D_PATH))

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PyTorch memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Groq LLM models (in order of preference)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# FLUX model configuration
DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
FLUX_IMAGE_SIZE = 1024
FLUX_FALLBACK_SIZE = 768


def print_device_info():
    """Print GPU/device information."""
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_colab_secrets():
    """Load API keys from Colab userdata if available."""
    try:
        from google.colab import userdata
        for key in ["HF_TOKEN", "GROQ_API_KEY"]:
            try:
                v = userdata.get(key)
                if v and not os.environ.get(key):
                    os.environ[key] = v
            except Exception:
                pass
    except ImportError:
        pass
