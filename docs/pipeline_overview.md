# Bodies of Flora - Pipeline Architecture

## Overview

Bodies of Flora is a multi-stage pipeline that converts natural language plant descriptions into 3D botanical models. The system leverages LLMs for species identification, diffusion models for 2D image generation, and 3D reconstruction models for mesh generation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│  "little cranberry-like plant with a vessel" / "Sarracenia purpurea" / etc  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALIAS NORMALIZATION                                  │
│  Maps indigenous/descriptive names to scientific names                       │
│  e.g., "miskominagaawanzh" → "Sarracenia purpurea"                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1a: SPECIES IDENTIFICATION                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Groq LLM (llama-3.3-70b-versatile)                                 │    │
│  │  - Comparative reasoning protocol                                    │    │
│  │  - Ethnobotanical knowledge                                          │    │
│  │  - Chain-of-thought decomposition                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: species_name, common_name, family, morphology, confidence          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1b: BOTANICAL ENRICHMENT                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Groq LLM (llama-3.3-70b-versatile)                                 │    │
│  │  - Expands morphology into visual description                        │    │
│  │  - Converts botanical jargon to plain language                       │    │
│  │  - Generates FLUX-optimized prompt                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: flux_prompt, clip_hint, negative_prompt                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2a: 2D IMAGE GENERATION                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FLUX.1-schnell (black-forest-labs)                                 │    │
│  │  - CPU offloading for memory efficiency                              │    │
│  │  - 1024×1024 output (768×768 fallback)                               │    │
│  │  - Dual prompt encoding (CLIP + T5)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: flux_*.png (photorealistic botanical render)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2b: IMAGE CLEANING                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4-Pass Pipeline:                                                    │    │
│  │  1. rembg - Background removal (U²-Net)                              │    │
│  │  2. Pixel kill - Remove white/black bleed artifacts                  │    │
│  │  3. Morphological ops - Close holes, remove noise                    │    │
│  │  4. Crop + center - 1024×1024 white canvas                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: clean_*.png (reconstruction-ready image)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │   MEMORY MANAGEMENT   │
                          │   Unload FLUX → GPU   │
                          │   free for Hunyuan3D  │
                          └───────────┬───────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: 3D MESH GENERATION                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Hunyuan3D-2 (Tencent)                                              │    │
│  │  - DiT-based image-to-3D                                             │    │
│  │  - FlashVDM acceleration (when available)                            │    │
│  │  - 50 inference steps, octree resolution 384                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Mesh Post-processing:                                               │    │
│  │  - Remove degenerate faces/vertices                                  │    │
│  │  - Fix normals                                                       │    │
│  │  - Remove thin slab artifacts                                        │    │
│  │  - Normalize to unit cube                                            │    │
│  │  - Laplacian smoothing                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: species_name_*.glb (textured 3D mesh)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUTS                                         │
│  - JSON: Full pipeline metadata + reasoning                                  │
│  - PNG: Raw FLUX image                                                       │
│  - PNG: Cleaned reconstruction image                                         │
│  - GLB: Textured 3D mesh                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Reference

### `src/config.py`

Configuration and environment setup:
- `DEVICE`: Automatically detects CUDA/CPU
- `OUTPUT_DIR`: Configurable via `BOF_OUTPUT_DIR` env var
- `HUNYUAN3D_PATH`: Configurable via `HUNYUAN3D_PATH` env var
- `GROQ_MODELS`: Ordered list of fallback models

### `src/alias_map.py`

Indigenous and descriptive name normalization:
- `INDIGENOUS_ALIAS_MAP`: Dictionary mapping names to scientific species
- `normalize_indigenous_input()`: Applies alias mapping before NLP

### `src/stage1_nlp.py`

Species identification and enrichment:
- `stage1a_nlp()`: LLM-based species identification with comparative reasoning
- `stage1b_enrich()`: Generates FLUX prompts from morphology

### `src/stage2_image.py`

2D image generation and cleaning:
- `stage2a_flux()`: FLUX.1-schnell image generation
- `stage2b_clean()`: 4-pass image cleaning pipeline
- `unload_flux_for_hunyuan()`: Memory management

### `src/stage3_mesh.py`

3D mesh generation:
- `stage3_hunyuan()`: Hunyuan3D-2 mesh generation + post-processing

### `src/pipeline.py`

Main orchestration:
- `run()`: Executes full pipeline, returns JSON + file paths

### `app/gradio_ui.py`

Web interface:
- `create_ui()`: Builds Gradio Blocks interface
- `main()`: Entry point for local execution

## Memory Management

The pipeline uses aggressive memory management to fit on 16GB+ GPUs:

1. **FLUX CPU Offloading**: Uses `enable_model_cpu_offload()` instead of `.to(device)` to keep model weights on CPU and only load layers during inference
2. **Explicit Unloading**: FLUX is fully unloaded before Hunyuan3D loads
3. **OOM Fallback**: FLUX falls back to 768×768 if 1024×1024 OOMs
4. **Garbage Collection**: Explicit `gc.collect()` + `torch.cuda.empty_cache()` between stages

## Comparative Reasoning Protocol

The Stage 1a prompt uses a specialized reasoning protocol for ambiguous inputs:

1. **Decompose** descriptive phrases into morphological/ecological clues
2. **Cross-reference** clues to find plants with ALL features
3. **Avoid literal matching** (e.g., "cranberry-like" ≠ cranberry)

Example:
```
Input: "little cranberry-like plant with a vessel"

Reasoning:
- "cranberry-like" = small, grows in bogs (habitat clue)
- "vessel" = pitcher-shaped structure (morphology clue)
- Small bog plant + pitcher = Sarracenia purpurea
- NOT Vaccinium (cranberry has no pitcher)
```

## Error Handling

- **Groq fallback**: Tries multiple models (llama-3.3-70b → llama-3.1-8b → gemma2-9b)
- **JSON extraction**: Handles markdown fences, partial JSON
- **OOM recovery**: Automatic resolution reduction
- **Mesh cleanup fallback**: Continues with partial cleanup on errors

## Performance Notes

| Stage | Time (A100) | VRAM Usage |
|-------|-------------|------------|
| 1a + 1b | ~2-3s | Minimal (API calls) |
| 2a FLUX | ~100-180s | ~12GB (offloaded) |
| 2b Clean | ~1-60s | ~2GB |
| 3 Hunyuan | ~50-70s | ~16GB |

Total pipeline: ~3-5 minutes per plant on A100.
