"""
Bodies of Flora - Indigenous Alias Map

Maps indigenous, descriptive, and common plant names to scientific names.
This enables the pipeline to handle culturally diverse plant references
including Ojibwe (Anishinaabe) names and comparative descriptions.
"""

INDIGENOUS_ALIAS_MAP = {
    # Ojibwe (Anishinaabe)
    "miskominagaawanzh": "Sarracenia purpurea",
    "little cranberry-like plant with a vessel": "Sarracenia purpurea",
    
    # Common descriptive patterns
    "bloodroot": "Sanguinaria canadensis",
    "the root that bleeds": "Sanguinaria canadensis",
    "root that bleeds": "Sanguinaria canadensis",
    "blood root": "Sanguinaria canadensis",
    "pitcher plant": "Sarracenia purpurea",
    "plant that catches flies": "Drosera rotundifolia",
    "fly trap": "Dionaea muscipula",
    "venus flytrap": "Dionaea muscipula",
    "thunder plant": "Podophyllum peltatum",
    "may apple": "Podophyllum peltatum",
    "snake root": "Aristolochia serpentaria",
    "indian pipe": "Monotropa uniflora",
    "ghost plant": "Monotropa uniflora",
}


def normalize_indigenous_input(text: str) -> str:
    """
    Check if input matches a known indigenous/descriptive alias.
    
    Returns the scientific name with original description if matched,
    otherwise returns the original text unchanged.
    
    Args:
        text: User input text describing a plant
        
    Returns:
        Scientific name with attribution if matched, or original text
    """
    raw = text.strip().lower()
    for pattern, species in INDIGENOUS_ALIAS_MAP.items():
        if pattern in raw:
            return f"{species} — originally described as: {text.strip()}"
    return text
