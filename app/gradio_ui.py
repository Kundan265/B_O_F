"""
Bodies of Flora - Gradio Web Interface

Interactive web UI for the NLP → 2D → 3D botanical generation pipeline.
"""

import os
from pathlib import Path

import gradio as gr

from src.pipeline import run
from src.config import load_colab_secrets, print_device_info


def _ui(text: str, groq_key: str, hf_token: str, seed: int):
    """
    Gradio UI callback function.
    
    Args:
        text: Plant description input
        groq_key: Groq API key
        hf_token: HuggingFace token
        seed: Random seed
        
    Returns:
        Tuple of outputs for Gradio components
    """
    if not text.strip():
        raise gr.Error("Enter a plant description")
    if not groq_key.strip():
        raise gr.Error("Groq API key required")

    nlp_json, img, clean, glb, log_text = run(
        text, groq_key, hf_token, int(seed)
    )

    # Build gallery
    gallery = []
    if img and os.path.exists(img):
        gallery.append((img, "FLUX 2D"))
    if clean and os.path.exists(clean):
        gallery.append((clean, "Cleaned"))

    # Collect downloadable files
    files = [f for f in [img, clean, glb] if f and os.path.exists(f)]
    
    return nlp_json, gallery, glb, files, log_text


def create_ui() -> gr.Blocks:
    """
    Create the Gradio Blocks interface.
    
    Returns:
        Configured Gradio Blocks app
    """
    with gr.Blocks(title="Bodies of Flora") as demo:
        gr.Markdown("""
# Bodies of Flora

Transform natural language plant descriptions into 3D botanical models.

**Pipeline:** NLP Species ID → FLUX 2D Image → Hunyuan3D-2 Mesh
        """)

        text_in = gr.Textbox(
            label="Plant Input",
            lines=8,
            placeholder=(
                "Describe a plant using scientific names, common names, "
                "indigenous names, or descriptive features.\n\n"
                "Examples:\n"
                "• Helonias bullata (Swamp Pink)\n"
                "• little cranberry-like plant with a vessel\n"
                "• the root that bleeds red"
            )
        )

        run_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Accordion("Configuration", open=False):
            groq_key = gr.Textbox(
                label="Groq API Key",
                type="password",
                value=os.environ.get("GROQ_API_KEY", "")
            )
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                type="password",
                value=os.environ.get("HF_TOKEN", "")
            )
            seed_sl = gr.Slider(
                0, 999999,
                value=42,
                step=1,
                label="Seed"
            )

        with gr.Tabs():
            with gr.Tab("NLP Output"):
                nlp_out = gr.Code(
                    label="Species + reasoning + enrichment + prompts",
                    language="json"
                )
            with gr.Tab("2D Images"):
                gallery = gr.Gallery(
                    label="FLUX → Cleaned",
                    columns=2,
                    height=420
                )
            with gr.Tab("3D Model"):
                model3d = gr.Model3D(
                    label="3D mesh (Hunyuan3D-2 local)",
                    clear_color=[0.85, 0.85, 0.85, 1.0]
                )

        files_out = gr.Files(label="Downloads")
        log_out = gr.Textbox(label="Pipeline Log", lines=25)

        run_btn.click(
            fn=_ui,
            inputs=[text_in, groq_key, hf_token, seed_sl],
            outputs=[nlp_out, gallery, model3d, files_out, log_out]
        )

    return demo


def main():
    """Launch the Gradio app."""
    # Load secrets if running in Colab
    load_colab_secrets()
    
    # Print device info
    print_device_info()
    
    # Create and launch UI
    demo = create_ui()
    demo.queue(max_size=10).launch(
        share=False,  # Set to True for public URL
        debug=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
