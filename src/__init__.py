"""
Bodies of Flora

A multi-stage pipeline for generating 3D botanical models from natural language
plant descriptions. Supports scientific names, common names, indigenous names,
and descriptive references.

Pipeline:
    1. NLP species identification (Groq LLM)
    2. FLUX.1-schnell 2D image generation
    3. Hunyuan3D-2 3D mesh generation
"""

__version__ = "0.1.0"
__author__ = "Bodies of Flora Contributors"

from .config import (
    DEVICE,
    OUTPUT_DIR,
    HUNYUAN3D_PATH,
    print_device_info,
    load_colab_secrets,
)
from .alias_map import (
    INDIGENOUS_ALIAS_MAP,
    normalize_indigenous_input,
)
from .stage1_nlp import (
    stage1a_nlp,
    stage1b_enrich,
)
from .stage2_image import (
    stage2a_flux,
    stage2b_clean,
    unload_flux_for_hunyuan,
)
from .stage3_mesh import (
    stage3_hunyuan,
)
from .pipeline import (
    run,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "DEVICE",
    "OUTPUT_DIR",
    "HUNYUAN3D_PATH",
    "print_device_info",
    "load_colab_secrets",
    # Alias map
    "INDIGENOUS_ALIAS_MAP",
    "normalize_indigenous_input",
    # Stage 1
    "stage1a_nlp",
    "stage1b_enrich",
    # Stage 2
    "stage2a_flux",
    "stage2b_clean",
    "unload_flux_for_hunyuan",
    # Stage 3
    "stage3_hunyuan",
    # Pipeline
    "run",
]
