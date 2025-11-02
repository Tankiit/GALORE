#!/usr/bin/env python3
"""
Interactive Text-to-Image and Image-to-Text Retrieval Demo

Explore your trained CLIP model's retrieval capabilities interactively!

Features:
- Text-to-Image: Enter text, see matching images
- Image-to-Text: Upload image, see matching captions
- Visualize similarity scores
- Compare MODE vs baseline models

Usage:
    # Web interface (recommended)
    python interactive_retrieval_demo.py --interface web --model_path ./mode_output/final_clip_model.pt

    # Command line interface
    python interactive_retrieval_demo.py --interface cli --model_path ./mode_output/final_clip_model.pt
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try importing transformers and gradio
try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")
    HAS_TRANSFORMERS = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    print("Info: gradio not installed. Web interface unavailable.")
    print("Install with: pip install gradio")
    HAS_GRADIO = False


# ============================================================================
# Core Retrieval Engine
# ============================================================================

class InteractiveRetrievalEngine:
    """
    Interactive retrieval system for exploring CLIP model capabilities.

    Supports:
    - Text-to-Image retrieval
    - Image-to-Text retrieval
    - Similarity visualization
    - Multi-modal search
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        device: str = 'auto'
    ):
        """
        Initialize retrieval engine.

        Args:
            model_path: Path to trained CLIP model (optional)
            clip_model_name: HuggingFace model name (if model_path not provided)
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load CLIP model
        if model_path and Path(model_path).exists():
            print(f"Loading trained model from: {model_path}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            state_dict = torch.load(model_path, map_location='cpu')
            self.clip_model.load_state_dict(state_dict)
        else:
            print(f"Loading pretrained model: {clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Storage for database of images/captions
        self.image_database = []
        self.caption_database = []
        self.image_embeddings = None
        self.caption_embeddings = None

    def load_database(
        self,
        images: Optional[List] = None,
        captions: Optional[List[str]] = None,
        database_path: Optional[str] = None
    ):
        """
        Load database of images and captions for retrieval.

        Args:
            images: List of PIL Images or image paths
            captions: List of caption strings
            database_path: Path to pre-computed embeddings (optional)
        """
        if database_path and Path(database_path).exists():
            print(f"Loading pre-computed embeddings from {database_path}")
            data = torch.load(database_path)
            self.image_embeddings = data['image_embeddings'].to(self.device)
            self.caption_embeddings = data['caption_embeddings'].to(self.device)
            self.image_database = data.get('images', [])
            self.caption_database = data.get('captions', [])
            print(f"Loaded {len(self.image_database)} images, {len(self.caption_database)} captions")
            return

        # Compute embeddings from scratch
        if images:
            print(f"Encoding {len(images)} images...")
            self.image_database = images
            self.image_embeddings = self._encode_images(images)

        if captions:
            print(f"Encoding {len(captions)} captions...")
            self.caption_database = captions
            self.caption_embeddings = self._encode_texts(captions)

    def _encode_images(self, images: List) -> torch.Tensor:
        """Encode images to embeddings"""
        embeddings = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]

                # Handle PIL Images or paths
                processed_imgs = []
                for img in batch:
                    if isinstance(img, str):
                        img = Image.open(img).convert('RGB')
                    processed_imgs.append(img)

                inputs = self.processor(
                    images=processed_imgs,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                features = self.clip_model.get_image_features(**inputs)
                features = F.normalize(features, dim=-1)
                embeddings.append(features.cpu())

        return torch.cat(embeddings, dim=0).to(self.device)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        embeddings = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                features = self.clip_model.get_text_features(**inputs)
                features = F.normalize(features, dim=-1)
                embeddings.append(features.cpu())

        return torch.cat(embeddings, dim=0).to(self.device)

    def text_to_image(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """
        Text-to-Image retrieval.

        Args:
            query_text: Text query
            top_k: Number of results to return

        Returns:
            indices: Top-K image indices
            scores: Similarity scores
        """
        if self.image_embeddings is None:
            raise ValueError("No image database loaded!")

        # Encode query text
        with torch.no_grad():
            inputs = self.processor(
                text=[query_text],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            text_emb = self.clip_model.get_text_features(**inputs)
            text_emb = F.normalize(text_emb, dim=-1)

        # Compute similarities
        similarities = (text_emb @ self.image_embeddings.T).squeeze(0)

        # Get top-K
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, top_k)

        return indices.cpu().tolist(), scores.cpu().tolist()

    def image_to_text(
        self,
        query_image,
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """
        Image-to-Text retrieval.

        Args:
            query_image: PIL Image or image path
            top_k: Number of results to return

        Returns:
            indices: Top-K caption indices
            scores: Similarity scores
        """
        if self.caption_embeddings is None:
            raise ValueError("No caption database loaded!")

        # Load image if path
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')

        # Encode query image
        with torch.no_grad():
            inputs = self.processor(
                images=[query_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            img_emb = self.clip_model.get_image_features(**inputs)
            img_emb = F.normalize(img_emb, dim=-1)

        # Compute similarities
        similarities = (img_emb @ self.caption_embeddings.T).squeeze(0)

        # Get top-K
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, top_k)

        return indices.cpu().tolist(), scores.cpu().tolist()

    def visualize_t2i_results(
        self,
        query_text: str,
        top_k: int = 5,
        save_path: Optional[str] = None
    ):
        """Visualize Text-to-Image retrieval results"""
        indices, scores = self.text_to_image(query_text, top_k)

        fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 5))
        if top_k == 1:
            axes = [axes]

        fig.suptitle(f'Text-to-Image: "{query_text}"', fontsize=16, fontweight='bold')

        for i, (idx, score) in enumerate(zip(indices, scores)):
            ax = axes[i]

            if idx < len(self.image_database):
                img = self.image_database[idx]
                if isinstance(img, str):
                    img = Image.open(img)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Image N/A', ha='center', va='center')

            ax.set_title(f'Rank {i+1}\nScore: {score:.3f}', fontsize=12)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        else:
            plt.show()

        plt.close()
        return fig

    def visualize_i2t_results(
        self,
        query_image,
        top_k: int = 5,
        save_path: Optional[str] = None
    ):
        """Visualize Image-to-Text retrieval results"""
        indices, scores = self.image_to_text(query_image, top_k)

        # Load image if path
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')

        fig = plt.figure(figsize=(12, 8))

        # Plot query image on left
        ax_img = plt.subplot(1, 2, 1)
        ax_img.imshow(query_image)
        ax_img.set_title('Query Image', fontsize=14, fontweight='bold')
        ax_img.axis('off')

        # Plot retrieved captions on right
        ax_text = plt.subplot(1, 2, 2)
        ax_text.axis('off')

        # Create text display
        y_pos = 0.95
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_k))

        ax_text.text(0.5, y_pos, 'Top Retrieved Captions',
                    fontsize=14, fontweight='bold', ha='center', va='top')
        y_pos -= 0.1

        for i, (idx, score) in enumerate(zip(indices, scores)):
            if idx < len(self.caption_database):
                caption = self.caption_database[idx]

                # Wrap long captions
                max_chars = 60
                if len(caption) > max_chars:
                    caption = caption[:max_chars-3] + '...'

                # Draw colored box
                rect = mpatches.FancyBboxPatch(
                    (0.05, y_pos - 0.08), 0.9, 0.08,
                    boxstyle="round,pad=0.01",
                    facecolor=colors[i], alpha=0.3,
                    edgecolor='black', linewidth=1.5
                )
                ax_text.add_patch(rect)

                # Draw text
                text = f"{i+1}. [{score:.3f}] {caption}"
                ax_text.text(0.5, y_pos - 0.04, text,
                           fontsize=11, ha='center', va='center',
                           wrap=True)

                y_pos -= 0.12

        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        else:
            plt.show()

        plt.close()
        return fig


# ============================================================================
# Web Interface (Gradio)
# ============================================================================

def create_web_interface(engine: InteractiveRetrievalEngine):
    """Create Gradio web interface"""

    if not HAS_GRADIO:
        print("Error: gradio not installed. Install with: pip install gradio")
        return None

    # Text-to-Image tab
    def t2i_search(query_text, top_k):
        if not query_text.strip():
            return None, "Please enter a text query"

        try:
            indices, scores = engine.text_to_image(query_text, int(top_k))

            # Create result display
            results = []
            for i, (idx, score) in enumerate(zip(indices, scores)):
                if idx < len(engine.image_database):
                    img = engine.image_database[idx]
                    if isinstance(img, str):
                        img = Image.open(img)
                    results.append((img, f"Rank {i+1}: {score:.3f}"))

            # Create visualization
            fig = engine.visualize_t2i_results(query_text, int(top_k))

            return fig, f"Found {len(results)} results"

        except Exception as e:
            return None, f"Error: {str(e)}"

    # Image-to-Text tab
    def i2t_search(query_image, top_k):
        if query_image is None:
            return None, "Please upload an image"

        try:
            indices, scores = engine.image_to_text(query_image, int(top_k))

            # Create result text
            result_text = "Top Retrieved Captions:\n\n"
            for i, (idx, score) in enumerate(zip(indices, scores)):
                if idx < len(engine.caption_database):
                    caption = engine.caption_database[idx]
                    result_text += f"{i+1}. [{score:.3f}] {caption}\n\n"

            # Create visualization
            fig = engine.visualize_i2t_results(query_image, int(top_k))

            return fig, result_text

        except Exception as e:
            return None, f"Error: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="Interactive Retrieval Demo") as demo:
        gr.Markdown("# 🔍 Interactive CLIP Retrieval Demo")
        gr.Markdown("Explore your trained model's text-to-image and image-to-text retrieval!")

        with gr.Tabs():
            # Text-to-Image Tab
            with gr.Tab("📝 Text → Image"):
                gr.Markdown("### Enter text to find matching images")

                with gr.Row():
                    with gr.Column():
                        t2i_text = gr.Textbox(
                            label="Text Query",
                            placeholder="e.g., 'a dog playing in a park'",
                            lines=2
                        )
                        t2i_topk = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of Results"
                        )
                        t2i_button = gr.Button("🔍 Search", variant="primary")

                    with gr.Column():
                        t2i_output_img = gr.Plot(label="Results")
                        t2i_output_text = gr.Textbox(label="Status")

                t2i_button.click(
                    fn=t2i_search,
                    inputs=[t2i_text, t2i_topk],
                    outputs=[t2i_output_img, t2i_output_text]
                )

                # Examples
                gr.Examples(
                    examples=[
                        ["a photo of a cat", 5],
                        ["a beautiful sunset over mountains", 5],
                        ["people playing soccer", 5],
                        ["a delicious pizza", 3],
                    ],
                    inputs=[t2i_text, t2i_topk]
                )

            # Image-to-Text Tab
            with gr.Tab("🖼️ Image → Text"):
                gr.Markdown("### Upload an image to find matching captions")

                with gr.Row():
                    with gr.Column():
                        i2t_image = gr.Image(
                            type="pil",
                            label="Upload Image"
                        )
                        i2t_topk = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of Results"
                        )
                        i2t_button = gr.Button("🔍 Search", variant="primary")

                    with gr.Column():
                        i2t_output_img = gr.Plot(label="Visualization")
                        i2t_output_text = gr.Textbox(
                            label="Retrieved Captions",
                            lines=10
                        )

                i2t_button.click(
                    fn=i2t_search,
                    inputs=[i2t_image, i2t_topk],
                    outputs=[i2t_output_img, i2t_output_text]
                )

        gr.Markdown("""
        ### 📊 About This Demo

        - **Text-to-Image**: Find images that match your text description
        - **Image-to-Text**: Find captions that describe your image
        - Powered by your trained MODE CLIP model
        - Scores show similarity (higher = better match)
        """)

    return demo


# ============================================================================
# Command Line Interface
# ============================================================================

class CommandLineInterface:
    """Interactive command-line interface"""

    def __init__(self, engine: InteractiveRetrievalEngine):
        self.engine = engine

    def run(self):
        """Run interactive CLI"""
        print("\n" + "="*70)
        print("  Interactive CLIP Retrieval Demo - Command Line Interface")
        print("="*70)
        print("\nCommands:")
        print("  t2i <text>     - Text-to-Image search")
        print("  i2t <path>     - Image-to-Text search (provide image path)")
        print("  help           - Show this help")
        print("  quit           - Exit")
        print("="*70 + "\n")

        while True:
            try:
                command = input("\n> ").strip()

                if not command:
                    continue

                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if command.lower() == 'help':
                    self.show_help()
                    continue

                parts = command.split(maxsplit=1)
                if len(parts) < 2:
                    print("Error: Invalid command. Type 'help' for usage.")
                    continue

                cmd, query = parts

                if cmd.lower() == 't2i':
                    self.text_to_image_search(query)
                elif cmd.lower() == 'i2t':
                    self.image_to_text_search(query)
                else:
                    print(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def text_to_image_search(self, query: str):
        """Execute T2I search"""
        print(f"\nSearching for images matching: '{query}'")
        print("-" * 70)

        indices, scores = self.engine.text_to_image(query, top_k=5)

        print("\nTop 5 Results:")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            print(f"  {i+1}. Image #{idx:04d} - Score: {score:.4f}")
            if idx < len(self.engine.caption_database):
                caption = self.engine.caption_database[idx]
                print(f"     Caption: {caption[:80]}...")

        # Ask to visualize
        resp = input("\nVisualize results? (y/n): ").strip().lower()
        if resp == 'y':
            self.engine.visualize_t2i_results(query, top_k=5)

    def image_to_text_search(self, image_path: str):
        """Execute I2T search"""
        print(f"\nSearching for captions matching image: {image_path}")
        print("-" * 70)

        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return

        indices, scores = self.engine.image_to_text(image_path, top_k=5)

        print("\nTop 5 Captions:")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            if idx < len(self.engine.caption_database):
                caption = self.engine.caption_database[idx]
                print(f"  {i+1}. [Score: {score:.4f}] {caption}")

        # Ask to visualize
        resp = input("\nVisualize results? (y/n): ").strip().lower()
        if resp == 'y':
            self.engine.visualize_i2t_results(image_path, top_k=5)

    def show_help(self):
        """Show help message"""
        print("\n" + "="*70)
        print("HELP")
        print("="*70)
        print("\nText-to-Image Search:")
        print("  t2i a photo of a dog playing")
        print("  t2i beautiful sunset over mountains")
        print("\nImage-to-Text Search:")
        print("  i2t /path/to/image.jpg")
        print("  i2t ./test_images/cat.png")
        print("\nOther:")
        print("  help - Show this help")
        print("  quit - Exit the program")
        print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive CLIP Retrieval Demo")
    parser.add_argument('--interface', type=str, default='web', choices=['web', 'cli'],
                       help='Interface type: web (Gradio) or cli (command line)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained CLIP model')
    parser.add_argument('--database_path', type=str, default=None,
                       help='Path to pre-computed embeddings database')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for web interface')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: transformers not installed!")
        print("Install with: pip install transformers torch")
        return

    print("\n" + "="*70)
    print("  Initializing Interactive Retrieval Engine")
    print("="*70 + "\n")

    # Create engine
    engine = InteractiveRetrievalEngine(
        model_path=args.model_path,
        device=args.device
    )

    # Load database
    if args.database_path:
        engine.load_database(database_path=args.database_path)
    else:
        print("\nWarning: No database loaded!")
        print("You need to provide image/caption database for retrieval.")
        print("Options:")
        print("  1. Use --database_path to load pre-computed embeddings")
        print("  2. Call engine.load_database(images, captions) in your code")
        print("\nFor demo purposes, creating sample database...")

        # Create dummy database for demo
        sample_captions = [
            "a photo of a cat sitting on a windowsill",
            "a beautiful sunset over the ocean",
            "people playing soccer in a park",
            "a delicious pizza with many toppings",
            "a modern city skyline at night",
            "a cute puppy playing with a ball",
            "mountains covered in snow",
            "a colorful flower garden",
            "a vintage car on a street",
            "children playing on a playground"
        ]
        engine.caption_database = sample_captions
        engine.caption_embeddings = engine._encode_texts(sample_captions)
        print(f"Created sample database with {len(sample_captions)} captions")

    # Launch interface
    if args.interface == 'web':
        if not HAS_GRADIO:
            print("\nError: Gradio not installed!")
            print("Install with: pip install gradio")
            print("Falling back to CLI interface...")
            args.interface = 'cli'
        else:
            print(f"\nLaunching web interface on port {args.port}...")
            demo = create_web_interface(engine)
            demo.launch(server_port=args.port, share=False)
            return

    if args.interface == 'cli':
        cli = CommandLineInterface(engine)
        cli.run()


if __name__ == '__main__':
    main()
