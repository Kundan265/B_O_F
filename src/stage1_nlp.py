"""
Bodies of Flora - Stage 1: NLP Processing

Species identification and botanical enrichment using Groq LLM.
- Stage 1a: Species identification with chain-of-thought comparative reasoning
- Stage 1b: Botanical enrichment and FLUX prompt generation
"""

import json
import re
import time
from typing import Dict, Any, Tuple

from groq import Groq

from .config import GROQ_MODELS


# Stage 1a system prompt with comparative reasoning protocol
STAGE1A_SYSTEM = """You are an expert ethnobotanist, taxonomic botanist, and historical flora interpreter.

CRITICAL — READ THIS FIRST:
Many plant descriptions use COMPARATIVE language where each word is a CLUE,
not a literal species reference. You MUST reason through these clues.

COMPARATIVE REASONING PROTOCOL (do this BEFORE identifying):
When input contains words like "-like", "similar to", "resembling", or descriptive
features ("with a vessel", "with sticky leaves", "with umbrella leaves"):

1. DECOMPOSE each phrase:
   - Size/habit comparisons ("-like", "small", "little") = ecological niche clue
   - Structural features ("vessel", "pitcher", "trap", "pouch", "cup") = morphology clue
   - Color references ("bleeds red", "white sap") = chemical/visual clue
   - Habitat hints ("bog", "marsh", "swamp") = ecological clue

2. CROSS-REFERENCE the clues:
   - What plant has ALL these features simultaneously?
   - The comparison word is usually about SIZE/HABITAT, not the species itself

3. EXAMPLES of correct reasoning:
   - "little cranberry-like plant with a vessel"
     → "cranberry-like" = small, grows in bogs (habitat clue, NOT species)
     → "vessel" = pitcher-shaped structure (morphology clue)
     → Small bog plant + pitcher structure = Sarracenia purpurea
     → NOT Vaccinium (cranberry has no vessel/pitcher)

   - "plant with sticky leaves that catches flies"
     → sticky leaves + insect capture = Drosera (sundew)
     → NOT a random sticky-leaved plant

   - "the root that bleeds red"
     → red sap from root = Sanguinaria canadensis (bloodroot)

   - "little plant like a sundew but with snap traps"
     → sundew-like = small, carnivorous (ecological comparison)
     → snap traps = Dionaea muscipula (Venus flytrap)

   - "tree with leaves like feathers and pods like beans"
     → compound pinnate leaves + legume pods = Fabaceae family
     → Could be Gleditsia, Robinia, etc.

KEY VOCABULARY from indigenous/descriptive names:
- "vessel", "pitcher", "cup", "jug" → pitcher plant (Sarracenia, Nepenthes, Darlingtonia)
- "trap", "catch", "eat insects", "sticky" → carnivorous plant
- "bleeding", "blood", "red sap" → Sanguinaria, Chelidonium
- "umbrella leaves" → Podophyllum, Diphylleia
- "ghost", "corpse", "no leaves" → mycoheterotrophic (Monotropa)
- "-like" suffix = comparison for SIZE or HABITAT, rarely means same species

RULES:
- Show your reasoning chain in the "reasoning" field
- Prefer specific identification over generic guesses
- If description implies carnivorous plant, identify as carnivorous
- If user says "whole plant" → whole-plant interpretation
- If user says "only seed/flower/root/leaf/fruit" → honor that focus
- NEVER leave species_name empty

Return ONLY valid JSON:
{
  "species_name": "Scientific name — NEVER empty",
  "common_name": "common name",
  "family": "botanical family",
  "confidence": 0.0-1.0,
  "reasoning": ["step 1...", "step 2...", "step 3..."],
  "morphology": {
    "growth_form": "herb/shrub/tree/vine/rosette/etc",
    "flower": "description or empty",
    "leaf": "description or empty",
    "fruit_seed": "description or empty",
    "root": "description or empty",
    "stem": "description or empty",
    "special_structures": "pitchers/traps/tendrils/etc or empty",
    "habitat": "bog/forest/prairie/etc or empty"
  },
  "part_focus": "whole plant | flower | seed | root | leaf | fruit | unknown"
}"""


# Stage 1b system prompt template
STAGE1B_SYSTEM_TEMPLATE = """You are a botanical image prompt engineer.

The plant is identified as:
- Species: {species}
- Common name: {common}
- Family: {family}
- Part focus: {part_focus}

Morphology from Stage 1a:
{morph_text}

YOUR JOB: Build a FLUX image generation prompt that is:
1. Visually faithful to this SPECIFIC species — include its defining features
2. Scientifically accurate — correct leaf shape, flower structure, growth habit
3. Clear for 3D reconstruction — three-quarter angle, isolated specimen

RULES:
- DO NOT change the identified species
- DO NOT replace it with a similar-looking but different plant
- If the plant has special structures (pitchers, traps, tendrils, thorns),
  those MUST be the dominant visual features
- Describe part-by-part: stem, leaves, flowers, fruits, roots, special structures
- Convert botanical jargon to plain visual language:
  "lanceolate" → "narrow spear-shaped"
  "urceolate" → "small urn-shaped"
  "pinnate" → "feather-shaped with leaflets on both sides"
  "pubescent" → "covered in fine soft hairs"
  "stoloniferous" → "spreading by horizontal runners"
- One isolated specimen on clean white studio background
- 80-150 words in the flux_prompt

Return ONLY valid JSON:
{{
  "species_name": "{species}",
  "visual_summary": "2-sentence description of the plant's appearance",
  "clip_hint": "12-word visual summary for CLIP encoder",
  "flux_prompt": "Three-quarter angle photorealistic 3D render of ... [80-150 words]",
  "negative_prompt": "things to exclude from the image"
}}"""


def _call_groq(
    messages: list,
    groq_key: str,
    temperature: float = 0.2,
    max_tokens: int = 1400
) -> Tuple[str, str]:
    """
    Call Groq API with fallback across multiple models.
    
    Args:
        messages: List of message dicts for the chat completion
        groq_key: Groq API key
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        Tuple of (response text, model name used)
        
    Raises:
        RuntimeError: If all models fail
    """
    client = Groq(api_key=groq_key)
    last_err = None
    
    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            txt = resp.choices[0].message.content
            if txt:
                return txt.strip(), model
        except Exception as e:
            last_err = e
            continue
    
    raise RuntimeError(f"Groq failed across all models: {last_err}")


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown fences.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON found
    """
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strip markdown fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass
    
    # Find JSON object in text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError("No valid JSON found in model response")


def stage1a_nlp(user_text: str, groq_key: str) -> Dict[str, Any]:
    """
    Stage 1a: Identify species with chain-of-thought comparative reasoning.
    
    Args:
        user_text: User's plant description
        groq_key: Groq API key
        
    Returns:
        Dictionary with species identification and morphology
    """
    t0 = time.time()
    print("━━━ Stage 1a: Species identification ━━━")

    txt, model = _call_groq(
        [
            {"role": "system", "content": STAGE1A_SYSTEM},
            {"role": "user", "content": f"Identify this plant:\n\n{user_text}"}
        ],
        groq_key=groq_key,
        temperature=0.1,
        max_tokens=1400
    )

    data = _extract_json(txt)
    
    # Ensure required fields
    data.setdefault("species_name", "Unknown plant")
    data.setdefault("common_name", "")
    data.setdefault("family", "")
    data.setdefault("confidence", 0.5)
    data.setdefault("reasoning", [])
    data.setdefault("morphology", {})
    data.setdefault("part_focus", "whole plant")

    dt = time.time() - t0
    print(f"  Model: {model} ({dt:.1f}s)")
    print(f"  Species: {data['species_name']}")
    print(f"  Common: {data['common_name']}")
    print(f"  Confidence: {data['confidence']}")
    print(f"  Reasoning: {data['reasoning'][:2]}")
    
    return data


def stage1b_enrich(stage1a_output: Dict[str, Any], groq_key: str) -> Dict[str, Any]:
    """
    Stage 1b: Build FLUX prompt from identified species + morphology.
    
    Args:
        stage1a_output: Output from stage1a_nlp
        groq_key: Groq API key
        
    Returns:
        Dictionary with FLUX prompt and visual metadata
    """
    t0 = time.time()
    print("━━━ Stage 1b: Enrichment + FLUX prompt ━━━")

    species = stage1a_output.get("species_name", "Unknown plant")
    common = stage1a_output.get("common_name", "")
    family = stage1a_output.get("family", "")
    part_focus = stage1a_output.get("part_focus", "whole plant")
    morph = stage1a_output.get("morphology", {})

    # Build morphology text for the prompt
    morph_lines = []
    for key in ["growth_form", "flower", "leaf", "fruit_seed", "root",
                "stem", "special_structures", "habitat"]:
        val = morph.get(key, "")
        if val and isinstance(val, str) and val.strip():
            morph_lines.append(f"  {key}: {val}")
    morph_text = "\n".join(morph_lines) if morph_lines else "  (minimal — enrich from your knowledge)"

    system_prompt = STAGE1B_SYSTEM_TEMPLATE.format(
        species=species,
        common=common,
        family=family,
        part_focus=part_focus,
        morph_text=morph_text
    )

    user_prompt = (
        f"Build the FLUX prompt for {species}"
        + (f" ({common})" if common else "")
        + f", focus: {part_focus}"
    )

    try:
        txt, model = _call_groq(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            groq_key=groq_key,
            temperature=0.15,
            max_tokens=1200
        )
        data = _extract_json(txt)
        dt = time.time() - t0
        print(f"  Model: {model} ({dt:.1f}s)")
    except Exception as e:
        print(f"  ⚠ Enrichment failed: {e} — building fallback prompt")
        data = {}

    # Ensure all fields exist
    data.setdefault("species_name", species)
    data.setdefault("clip_hint", f"3D render {common or species} botanical specimen")

    if not data.get("negative_prompt"):
        data["negative_prompt"] = (
            "multiple plants, bouquet, pot, vase, soil, landscape, garden, "
            "text, watermark, hands, person, blurry, low quality"
        )

    # Generic fallback prompt using actual species
    if not data.get("flux_prompt") or len(data["flux_prompt"].split()) < 15:
        morph_desc = ", ".join(
            f"{k}: {v}" for k, v in morph.items()
            if v and isinstance(v, str) and v.strip()
        )
        data["flux_prompt"] = (
            f"Three-quarter angle photorealistic 3D render of {species}"
            + (f" ({common})" if common else "")
            + f". {morph_desc}. "
            + f"Single isolated botanical specimen, {part_focus}, "
            + "clean white studio background, volumetric lighting, "
            + "sharp botanical detail, museum-quality scientific rendering."
        )

    print(f"  Prompt: {data['flux_prompt'][:150]}...")
    return data
