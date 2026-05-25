"""
Bodies of Flora - Pipeline Orchestration

Main orchestration function that runs the full NLP → 2D → 3D pipeline.
"""

import json
import os
import traceback
from typing import Tuple, Optional, List

from .alias_map import normalize_indigenous_input
from .stage1_nlp import stage1a_nlp, stage1b_enrich
from .stage2_image import stage2a_flux, stage2b_clean, unload_flux_for_hunyuan
from .stage3_mesh import stage3_hunyuan


def run(
    user_text: str,
    groq_key: str,
    hf_token: str = "",
    seed: int = 42
) -> Tuple[str, Optional[str], Optional[str], Optional[str], str]:
    """
    Run the full Bodies of Flora pipeline.
    
    Pipeline stages:
    1a. Species identification (Groq LLM)
    1b. Botanical enrichment + FLUX prompt generation
    2a. FLUX.1-schnell 2D image generation
    2b. Image cleaning (background removal, morphological cleanup)
    3.  Hunyuan3D-2 3D mesh generation
    
    Args:
        user_text: Plant description from user
        groq_key: Groq API key
        hf_token: Optional HuggingFace token
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - JSON string with pipeline results
        - Raw image path (or None on error)
        - Cleaned image path (or None on error)
        - GLB mesh path (or None on error)
        - Log string
    """
    logs: List[str] = []

    def log(msg: str):
        print(msg)
        logs.append(str(msg))

    try:
        # Validate inputs
        if not user_text or not user_text.strip():
            raise ValueError("Empty plant input")
        if not groq_key or not groq_key.strip():
            raise ValueError("Missing GROQ_API_KEY")

        # Set environment variables
        os.environ["GROQ_API_KEY"] = groq_key.strip()
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()

        seed = int(seed)

        # Apply indigenous alias normalization BEFORE NLP
        normalized_text = normalize_indigenous_input(user_text)
        if normalized_text != user_text:
            log(f"  Indigenous alias matched: {user_text.strip()}")
            log(f"  → Normalized to: {normalized_text}")
        
        log(f"Input: {user_text[:100]}...")
        if normalized_text != user_text:
            log(f"Normalized: {normalized_text[:100]}...")

        # Stage 1a — Species ID with comparative reasoning
        log("\n▸ Stage 1a — Species identification")
        stage1a = stage1a_nlp(normalized_text, groq_key.strip())
        log(f"  → {stage1a['species_name']} ({stage1a.get('common_name', '')})")
        log(f"  Confidence: {stage1a.get('confidence', '?')}")
        for r in stage1a.get("reasoning", [])[:3]:
            log(f"  Reasoning: {r}")

        # Stage 1b — Enrichment + FLUX prompt
        log("\n▸ Stage 1b — Botanical enrichment + FLUX prompt")
        stage1b = stage1b_enrich(stage1a, groq_key.strip())

        flux_prompt = stage1b.get("flux_prompt", "")
        clip_hint = stage1b.get("clip_hint", "")
        if not flux_prompt:
            raise ValueError("Stage 1b returned no FLUX prompt")
        log(f"  Prompt: {flux_prompt[:150]}...")

        # Stage 2a — FLUX image
        log("\n▸ Stage 2a — FLUX image generation")
        img_path = stage2a_flux(
            flux_prompt,
            clip_hint=clip_hint,
            seed=seed,
            hf_token=hf_token
        )

        # Stage 2b — Clean
        log("\n▸ Stage 2b — Image cleaning")
        clean_path = stage2b_clean(img_path)

        # Free FLUX VRAM before Hunyuan3D
        unload_flux_for_hunyuan()

        # Stage 3 — Hunyuan3D local
        label = stage1a.get("species_name", "plant")
        log("\n▸ Stage 3 — Hunyuan3D-2 local mesh generation")
        glb_path = stage3_hunyuan(clean_path, seed=seed, label=label)

        # Build final result JSON
        final_json = {
            "input_text": user_text,
            "normalized_input": normalized_text if normalized_text != user_text else None,
            "stage1a_species_id": stage1a,
            "stage1b_enrichment": {
                k: v for k, v in stage1b.items()
                if k != "flux_prompt"  # Keep output compact
            },
            "flux_prompt": flux_prompt,
            "outputs": {
                "image_raw": img_path,
                "image_clean": clean_path,
                "mesh_glb": glb_path,
            },
            "seed": seed,
        }

        log("\n✓ Pipeline complete")
        return (
            json.dumps(final_json, indent=2, ensure_ascii=False),
            img_path,
            clean_path,
            glb_path,
            "\n".join(logs)
        )

    except Exception as e:
        tb = traceback.format_exc()
        logs.append(f"\n✗ ERROR: {e}\n{tb}")
        return (
            json.dumps({"error": str(e), "traceback": tb}, indent=2),
            None,
            None,
            None,
            "\n".join(logs)
        )
