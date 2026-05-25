"""
Bodies of Flora - Stage 3: 3D Mesh Generation

Hunyuan3D-2 local mesh generation from cleaned 2D images.
Includes mesh repair, normalization, and GLB export.
"""

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import DEVICE, OUTPUT_DIR


def stage3_hunyuan(
    image_path: str,
    seed: int = 42,
    label: str = "plant"
) -> str:
    """
    Generate 3D mesh locally with Hunyuan3D-2.
    
    Args:
        image_path: Path to cleaned input image
        seed: Random seed for reproducibility
        label: Label for output filename (species name)
        
    Returns:
        Path to generated GLB mesh file
    """
    t0 = time.time()
    print("━━━ Stage 3: Hunyuan3D-2 local ━━━")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    import trimesh as tm

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        torch_dtype=torch.float16,
        device=DEVICE
    )

    # Try to enable FlashVDM for faster generation
    try:
        pipe.enable_flashvdm(mc_algo='mc')
        print("  FlashVDM enabled")
    except Exception as e:
        print(f"  FlashVDM not available: {e}")

    image = Image.open(image_path).convert("RGB")
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))

    with torch.no_grad():
        mesh = pipe(
            image=image,
            num_inference_steps=50,
            guidance_scale=7.5,
            octree_resolution=384,
            num_chunks=200000,
            generator=gen,
            output_type="trimesh"
        )

    # Handle list output
    if isinstance(mesh, list):
        meshes = [m for m in mesh if isinstance(m, tm.Trimesh)]
        if not meshes:
            raise ValueError("Hunyuan3D returned no trimesh objects")
        mesh = meshes[0] if len(meshes) == 1 else tm.util.concatenate(meshes)

    # Mesh cleanup and repair
    mesh = _repair_mesh(mesh)

    # Generate safe filename and export
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") or "plant"
    glb_path = OUTPUT_DIR / f"{safe_label}_{int(time.time())}.glb"
    mesh.export(str(glb_path))
    
    print(f"  ✓ Mesh saved ({time.time() - t0:.1f}s): {glb_path}")
    return str(glb_path)


def _repair_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Apply mesh repair operations.
    
    - Remove degenerate faces
    - Remove unreferenced vertices
    - Fix normals
    - Remove thin slab artifacts
    - Normalize to unit cube
    - Apply light smoothing
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Repaired trimesh object
    """
    import trimesh as tm
    
    try:
        # Basic cleanup
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        if hasattr(mesh, 'remove_unreferenced_vertices'):
            mesh.remove_unreferenced_vertices()
        tm.repair.fix_normals(mesh)

        # Slab removal - filter out thin flat artifacts
        if hasattr(mesh, 'split'):
            components = mesh.split()
            if len(components) > 1:
                total_faces = sum(len(c.faces) for c in components)
                kept = []
                for c in components:
                    extents = sorted(c.bounding_box.extents)
                    # Keep if not too flat or has significant face count
                    aspect_ratio = extents[0] / max(extents[2], 1e-8)
                    face_ratio = len(c.faces) / total_faces
                    if aspect_ratio >= 0.08 or face_ratio > 0.5:
                        kept.append(c)
                if kept:
                    mesh = tm.util.concatenate(kept)

        # Normalize to unit cube centered at origin
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.vertices -= center
        scale = np.max(mesh.bounds[1] - mesh.bounds[0])
        if scale > 1e-8:
            mesh.vertices /= scale

        # Light Laplacian smoothing
        tm.smoothing.filter_laplacian(mesh, iterations=1)
        
    except Exception as e:
        print(f"  ⚠ Mesh cleanup partial: {e}")

    return mesh
