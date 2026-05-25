# Bodies of Flora

Transform natural language plant descriptions into 3D botanical models.

**Pipeline:** NLP Species Identification → FLUX 2D Image Generation → Hunyuan3D-2 3D Mesh

## Features

- **Multi-modal input**: Accepts scientific names, common names, indigenous names, and descriptive references
- **Indigenous knowledge support**: Includes Ojibwe (Anishinaabe) plant names and comparative descriptions
- **Chain-of-thought reasoning**: Uses LLM comparative reasoning to correctly identify plants from vague descriptions
- **End-to-end generation**: Produces 2D renders and 3D GLB meshes from text input
- **Interactive UI**: Gradio web interface for easy use

## Pipeline Stages

1. **Stage 1a - Species Identification**: Groq LLM identifies the plant species using ethnobotanical and comparative reasoning
2. **Stage 1b - Botanical Enrichment**: Generates detailed FLUX prompts based on plant morphology
3. **Stage 2a - 2D Generation**: FLUX.1-schnell creates photorealistic botanical renders
4. **Stage 2b - Image Cleaning**: Background removal and morphological cleanup for 3D reconstruction
5. **Stage 3 - 3D Generation**: Hunyuan3D-2 generates textured 3D mesh

## Quick Start (Google Colab)

The easiest way to run Bodies of Flora is in Google Colab with GPU:

1. Open `notebooks/Robust_pipeline.ipynb` in Colab
2. Set runtime to GPU (A100 recommended)
3. Run Cell 1 to install dependencies
4. **Restart runtime**
5. Run Cell 2 to launch the Gradio UI

## Local Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- [Groq API key](https://console.groq.com/) (required)
- [HuggingFace token](https://huggingface.co/settings/tokens) (optional, for gated models)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bodies-of-flora.git
cd bodies-of-flora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone Hunyuan3D-2
git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
pip install -r Hunyuan3D-2/requirements.txt

# Build custom rasterizer (optional, for texture generation)
cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../../..

# Set environment variables
export GROQ_API_KEY="your-groq-api-key"
export HF_TOKEN="your-huggingface-token"  # Optional
```

### Run the UI

```bash
python -m app.gradio_ui
```

### Programmatic Usage

```python
from src import run

result_json, img_path, clean_path, glb_path, logs = run(
    user_text="Helonias bullata (Swamp Pink)",
    groq_key="your-groq-api-key",
    hf_token="your-hf-token",  # Optional
    seed=42
)
```

## Example Inputs

The pipeline handles diverse input types:

```
# Scientific name
Sarracenia purpurea

# Common name
Swamp Pink

# Indigenous name (Ojibwe)
miskominagaawanzh

# Descriptive reference
little cranberry-like plant with a vessel

# Morphological description
the root that bleeds red
```

## Project Structure

```
bodies-of-flora/
├── src/
│   ├── __init__.py          # Package exports
│   ├── config.py             # Configuration
│   ├── alias_map.py          # Indigenous name mapping
│   ├── stage1_nlp.py         # Species ID + enrichment
│   ├── stage2_image.py       # FLUX + image cleaning
│   ├── stage3_mesh.py        # Hunyuan3D-2 mesh generation
│   └── pipeline.py           # Orchestration
├── app/
│   └── gradio_ui.py          # Web interface
├── notebooks/
│   └── Robust_pipeline.ipynb # Colab notebook
├── data/
│   ├── plant_texts/          # Sample inputs
│   └── research/             # Research database
├── docs/
│   └── pipeline_overview.md  # Architecture docs
├── requirements.txt
├── LICENSE
└── README.md
```

## Configuration

Environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM access |
| `HF_TOKEN` | No | HuggingFace token for model downloads |
| `HUNYUAN3D_PATH` | No | Path to Hunyuan3D-2 clone (default: `./Hunyuan3D-2`) |
| `BOF_OUTPUT_DIR` | No | Output directory (default: `./outputs`) |

## Indigenous Knowledge

This project includes indigenous plant names, particularly from Ojibwe (Anishinaabe) sources. These names encode generations of botanical knowledge about plant morphology, ecology, and traditional uses.

Example: **miskominagaawanzh** describes Sarracenia purpurea as a "little cranberry-like plant with a vessel" - encoding both habitat (bog, like cranberries) and morphology (pitcher-shaped leaves).

## Acknowledgments

- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) by Tencent
- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by Black Forest Labs
- [Groq](https://groq.com/) for fast LLM inference
- Indigenous botanical knowledge from Ojibwe and other First Nations sources

## License

MIT License - see [LICENSE](LICENSE) for details.

Note: This license applies to the code. Research data and indigenous knowledge may have additional considerations. Please see `data/README.md` for details.
