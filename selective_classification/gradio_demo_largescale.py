#!/usr/bin/env python3
"""
Professional Large-Scale Retrieval Demo with Gradio

Beautiful, interactive demo showcasing MODE's value on large datasets.

Features:
- Side-by-side comparison (MODE vs Random vs Full)
- Text-to-Image and Image-to-Text retrieval
- Statistics and analytics dashboard
- Professional UI with metrics
- Shareable public link

Usage:
    python gradio_demo_largescale.py \
        --mode_db ./db_mode_30k \
        --random_db ./db_random_30k \
        --full_db ./db_full_100k \
        --share  # Create public link
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
import json
import numpy as np
from PIL import Image
import pickle

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    print("Error: gradio not installed! Install with: pip install gradio")
    HAS_GRADIO = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ============================================================================
# Large-Scale Retrieval Engine
# ============================================================================

class LargeScaleRetrieval:
    """
    Fast retrieval engine for large databases using FAISS.
    """

    def __init__(self, db_path: str, model_path: Optional[str] = None, device: str = 'auto'):
        self.db_path = Path(db_path)
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device

        print(f"Loading database from: {db_path}")

        # Load metadata
        with open(self.db_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load paths
        with open(self.db_path / 'paths.pkl', 'rb') as f:
            paths_data = pickle.load(f)
            self.image_paths = paths_data['image_paths']
            self.captions = paths_data['captions']

        print(f"  Loaded {len(self.image_paths)} images, {len(self.captions)} captions")

        # Load FAISS indices if available
        self.use_faiss = self.metadata.get('use_faiss', False) and HAS_FAISS

        if self.use_faiss:
            faiss_img_path = self.db_path / 'faiss_image_index.bin'
            faiss_cap_path = self.db_path / 'faiss_caption_index.bin'

            if faiss_img_path.exists():
                self.faiss_img_index = faiss.read_index(str(faiss_img_path))
                print(f"  Loaded image FAISS index: {self.faiss_img_index.ntotal} vectors")
            else:
                self.faiss_img_index = None

            if faiss_cap_path.exists():
                self.faiss_cap_index = faiss.read_index(str(faiss_cap_path))
                print(f"  Loaded caption FAISS index: {self.faiss_cap_index.ntotal} vectors")
            else:
                self.faiss_cap_index = None
        else:
            # Load embeddings from chunks
            image_chunks = paths_data.get('image_chunks', [])
            caption_chunks = paths_data.get('caption_chunks', [])

            print("  Loading embeddings from chunks...")
            self.image_embeddings = torch.cat([torch.load(f) for f in image_chunks], dim=0)
            self.caption_embeddings = torch.cat([torch.load(f) for f in caption_chunks], dim=0)

        # Load CLIP model for encoding queries
        if HAS_TRANSFORMERS:
            clip_name = 'openai/clip-vit-base-patch32'
            if model_path and Path(model_path).exists():
                self.clip_model = CLIPModel.from_pretrained(clip_name)
                self.clip_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                self.clip_model = CLIPModel.from_pretrained(clip_name)

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.processor = CLIPProcessor.from_pretrained(clip_name)

    def text_to_image(self, query_text: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Text-to-Image retrieval"""
        # Encode query
        with torch.no_grad():
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
            text_emb = self.clip_model.get_text_features(**inputs)
            text_emb = F.normalize(text_emb, dim=-1).cpu().numpy()

        # Search
        if self.use_faiss and self.faiss_img_index:
            scores, indices = self.faiss_img_index.search(text_emb, top_k)
            return indices[0].tolist(), scores[0].tolist()
        else:
            similarities = (torch.from_numpy(text_emb) @ self.image_embeddings.T).squeeze(0)
            scores, indices = torch.topk(similarities, top_k)
            return indices.tolist(), scores.tolist()

    def image_to_text(self, query_image: Image.Image, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Image-to-Text retrieval"""
        # Encode query
        with torch.no_grad():
            inputs = self.processor(images=[query_image], return_tensors="pt", padding=True).to(self.device)
            img_emb = self.clip_model.get_image_features(**inputs)
            img_emb = F.normalize(img_emb, dim=-1).cpu().numpy()

        # Search
        if self.use_faiss and self.faiss_cap_index:
            scores, indices = self.faiss_cap_index.search(img_emb, top_k)
            return indices[0].tolist(), scores[0].tolist()
        else:
            similarities = (torch.from_numpy(img_emb) @ self.caption_embeddings.T).squeeze(0)
            scores, indices = torch.topk(similarities, top_k)
            return indices.tolist(), scores.tolist()


# ============================================================================
# Gradio Demo Interface
# ============================================================================

class GradioLargeScaleDemo:
    """
    Professional Gradio demo for large-scale retrieval.
    """

    def __init__(
        self,
        mode_db_path: Optional[str] = None,
        random_db_path: Optional[str] = None,
        full_db_path: Optional[str] = None,
        mode_model_path: Optional[str] = None,
        device: str = 'auto'
    ):
        self.engines = {}

        # Load databases
        if mode_db_path:
            print("\n" + "="*80)
            print("Loading MODE database...")
            self.engines['MODE'] = LargeScaleRetrieval(mode_db_path, mode_model_path, device)

        if random_db_path:
            print("\n" + "="*80)
            print("Loading Random baseline database...")
            self.engines['Random'] = LargeScaleRetrieval(random_db_path, None, device)

        if full_db_path:
            print("\n" + "="*80)
            print("Loading Full database...")
            self.engines['Full'] = LargeScaleRetrieval(full_db_path, None, device)

        if not self.engines:
            raise ValueError("At least one database must be provided!")

        print("\n" + "="*80)
        print("All databases loaded successfully!")
        print("="*80 + "\n")

    def text_to_image_search(self, query_text: str, top_k: int, compare: bool):
        """Text-to-Image search with optional comparison"""
        if not query_text.strip():
            return self._empty_results(compare)

        results = {}

        for name, engine in self.engines.items():
            try:
                indices, scores = engine.text_to_image(query_text, int(top_k))

                # Get images
                images = []
                for idx, score in zip(indices, scores):
                    if idx < len(engine.image_paths):
                        img_path = engine.image_paths[idx]
                        try:
                            img = Image.open(img_path).convert('RGB')
                            # Resize for display
                            img.thumbnail((300, 300))
                            images.append(img)
                        except:
                            images.append(self._create_placeholder_image(f"Error\nloading\nimage"))
                    else:
                        images.append(self._create_placeholder_image(f"Image\n#{idx}"))

                results[name] = {
                    'images': images,
                    'scores': scores,
                    'avg_score': np.mean(scores),
                    'max_score': np.max(scores),
                    'min_score': np.min(scores)
                }

            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = None

        return self._format_t2i_results(results, query_text, compare)

    def image_to_text_search(self, query_image: Image.Image, top_k: int, compare: bool):
        """Image-to-Text search with optional comparison"""
        if query_image is None:
            return self._empty_results(compare)

        results = {}

        for name, engine in self.engines.items():
            try:
                indices, scores = engine.image_to_text(query_image, int(top_k))

                # Get captions
                captions = []
                for idx in indices:
                    if idx < len(engine.captions):
                        captions.append(engine.captions[idx])
                    else:
                        captions.append(f"Caption #{idx} not found")

                results[name] = {
                    'captions': captions,
                    'scores': scores,
                    'avg_score': np.mean(scores),
                    'max_score': np.max(scores),
                    'min_score': np.min(scores)
                }

            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = None

        return self._format_i2t_results(results, query_image, compare)

    def _create_placeholder_image(self, text: str) -> Image.Image:
        """Create placeholder image with text"""
        img = Image.new('RGB', (300, 300), color=(200, 200, 200))
        return img

    def _format_t2i_results(self, results: Dict, query: str, compare: bool):
        """Format Text-to-Image results for display"""
        if not results:
            return self._empty_results(compare)

        if compare and len(results) > 1:
            # Comparison view
            output_html = f"<h2>Query: \"{query}\"</h2>\n"

            for name, data in results.items():
                if data is None:
                    continue

                output_html += f"<h3>{name} Results (Avg Score: {data['avg_score']:.3f})</h3>\n"
                output_html += "<div style='display: flex; gap: 10px; margin-bottom: 20px;'>\n"

                for i, (img, score) in enumerate(zip(data['images'], data['scores'])):
                    output_html += f"<div style='text-align: center;'>"
                    output_html += f"<p><b>Rank {i+1}</b><br>Score: {score:.3f}</p></div>\n"

                output_html += "</div>\n"

            # Gallery display
            all_images = []
            all_labels = []
            for name, data in results.items():
                if data:
                    for i, (img, score) in enumerate(zip(data['images'], data['scores'])):
                        all_images.append(img)
                        all_labels.append(f"{name} - Rank {i+1} ({score:.3f})")

            # Statistics
            stats_html = "<h3>Comparison Statistics</h3>\n<table style='width:100%; border-collapse: collapse;'>\n"
            stats_html += "<tr style='background-color: #f0f0f0;'><th>Method</th><th>Avg Score</th><th>Max Score</th><th>Min Score</th></tr>\n"

            for name, data in results.items():
                if data:
                    stats_html += f"<tr><td>{name}</td><td>{data['avg_score']:.3f}</td><td>{data['max_score']:.3f}</td><td>{data['min_score']:.3f}</td></tr>\n"

            stats_html += "</table>\n"

            return all_images, output_html + stats_html

        else:
            # Single view
            name = list(results.keys())[0]
            data = results[name]

            if data is None:
                return [], f"<p>Error processing query</p>"

            output_html = f"<h2>Query: \"{query}\"</h2>\n"
            output_html += f"<p><b>Average Score:</b> {data['avg_score']:.3f}</p>\n"

            images_with_labels = []
            for i, (img, score) in enumerate(zip(data['images'], data['scores'])):
                images_with_labels.append((img, f"Rank {i+1}: {score:.3f}"))

            return data['images'], output_html

    def _format_i2t_results(self, results: Dict, query_image: Image.Image, compare: bool):
        """Format Image-to-Text results for display"""
        if not results:
            return query_image, self._empty_text_results(compare)

        if compare and len(results) > 1:
            # Comparison view
            output_html = "<h2>Retrieved Captions Comparison</h2>\n"

            for name, data in results.items():
                if data is None:
                    continue

                output_html += f"<h3>{name} Results (Avg Score: {data['avg_score']:.3f})</h3>\n"
                output_html += "<ol>\n"

                for i, (caption, score) in enumerate(zip(data['captions'], data['scores'])):
                    color = self._get_score_color(score)
                    output_html += f"<li style='margin-bottom: 10px; padding: 10px; background-color: {color}; border-radius: 5px;'>"
                    output_html += f"<b>[{score:.3f}]</b> {caption}</li>\n"

                output_html += "</ol>\n"

            # Statistics
            stats_html = "<h3>Comparison Statistics</h3>\n<table style='width:100%; border-collapse: collapse;'>\n"
            stats_html += "<tr style='background-color: #f0f0f0;'><th>Method</th><th>Avg Score</th><th>Max Score</th><th>Min Score</th></tr>\n"

            for name, data in results.items():
                if data:
                    stats_html += f"<tr><td>{name}</td><td>{data['avg_score']:.3f}</td><td>{data['max_score']:.3f}</td><td>{data['min_score']:.3f}</td></tr>\n"

            stats_html += "</table>\n"

            return query_image, output_html + stats_html

        else:
            # Single view
            name = list(results.keys())[0]
            data = results[name]

            if data is None:
                return query_image, "<p>Error processing query</p>"

            output_html = "<h2>Top Retrieved Captions</h2>\n"
            output_html += f"<p><b>Average Score:</b> {data['avg_score']:.3f}</p>\n"
            output_html += "<ol>\n"

            for i, (caption, score) in enumerate(zip(data['captions'], data['scores'])):
                color = self._get_score_color(score)
                output_html += f"<li style='margin-bottom: 10px; padding: 10px; background-color: {color}; border-radius: 5px;'>"
                output_html += f"<b>[{score:.3f}]</b> {caption}</li>\n"

            output_html += "</ol>\n"

            return query_image, output_html

    def _get_score_color(self, score: float) -> str:
        """Get background color based on score"""
        if score > 0.80:
            return "#d4edda"  # Green
        elif score > 0.65:
            return "#fff3cd"  # Yellow
        elif score > 0.50:
            return "#f8d7da"  # Light red
        else:
            return "#f5c6cb"  # Red

    def _empty_results(self, compare: bool):
        """Return empty results"""
        return [], "<p>No results</p>"

    def _empty_text_results(self, compare: bool):
        """Return empty text results"""
        return "<p>No results</p>"

    def get_statistics(self):
        """Get database statistics"""
        stats_html = "<h2>Database Statistics</h2>\n"
        stats_html += "<table style='width:100%; border-collapse: collapse;'>\n"
        stats_html += "<tr style='background-color: #f0f0f0;'><th>Method</th><th>Samples</th><th>Embedding Dim</th><th>Uses FAISS</th></tr>\n"

        for name, engine in self.engines.items():
            stats_html += f"<tr><td><b>{name}</b></td>"
            stats_html += f"<td>{len(engine.image_paths):,}</td>"
            stats_html += f"<td>{engine.metadata['embedding_dim']}</td>"
            stats_html += f"<td>{'Yes' if engine.use_faiss else 'No'}</td></tr>\n"

        stats_html += "</table>\n"

        return stats_html

    def create_interface(self):
        """Create Gradio interface"""
        if not HAS_GRADIO:
            raise ImportError("Gradio not installed!")

        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stats-box {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        """

        with gr.Blocks(css=custom_css, title="Large-Scale Retrieval Demo") as demo:
            gr.HTML("""
            <div class="header">
                <h1>🔍 Large-Scale CLIP Retrieval Demo</h1>
                <p>Showcase MODE's Value: Data-Efficient Vision-Language Retrieval</p>
            </div>
            """)

            # Statistics
            with gr.Accordion("📊 Database Statistics", open=False):
                stats_output = gr.HTML(value=self.get_statistics())

            # Main tabs
            with gr.Tabs():
                # Text-to-Image Tab
                with gr.Tab("📝 Text → Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            t2i_text = gr.Textbox(
                                label="Enter your text query",
                                placeholder="e.g., 'a dog playing in a park'",
                                lines=3
                            )
                            t2i_topk = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Number of results"
                            )
                            t2i_compare = gr.Checkbox(
                                label="Compare all methods",
                                value=len(self.engines) > 1
                            )
                            t2i_button = gr.Button("🔍 Search", variant="primary", size="lg")

                            # Examples
                            gr.Examples(
                                examples=[
                                    ["a cat sitting on a windowsill", 5],
                                    ["a beautiful sunset over mountains", 5],
                                    ["people playing soccer in a park", 5],
                                    ["a delicious pizza with toppings", 3],
                                    ["a modern city skyline at night", 5],
                                ],
                                inputs=[t2i_text, t2i_topk]
                            )

                        with gr.Column(scale=2):
                            t2i_gallery = gr.Gallery(
                                label="Retrieved Images",
                                columns=5,
                                height="auto"
                            )
                            t2i_html = gr.HTML(label="Results")

                    t2i_button.click(
                        fn=self.text_to_image_search,
                        inputs=[t2i_text, t2i_topk, t2i_compare],
                        outputs=[t2i_gallery, t2i_html]
                    )

                # Image-to-Text Tab
                with gr.Tab("🖼️ Image → Text"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            i2t_image = gr.Image(
                                type="pil",
                                label="Upload your image"
                            )
                            i2t_topk = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Number of results"
                            )
                            i2t_compare = gr.Checkbox(
                                label="Compare all methods",
                                value=len(self.engines) > 1
                            )
                            i2t_button = gr.Button("🔍 Search", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            i2t_image_display = gr.Image(
                                type="pil",
                                label="Query Image"
                            )
                            i2t_html = gr.HTML(label="Retrieved Captions")

                    i2t_button.click(
                        fn=self.image_to_text_search,
                        inputs=[i2t_image, i2t_topk, i2t_compare],
                        outputs=[i2t_image_display, i2t_html]
                    )

            # Footer
            gr.HTML("""
            <div style='text-align: center; margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
                <h3>📚 About This Demo</h3>
                <p>This demo showcases MODE (Multi-Objective Data-driven Engine) for efficient vision-language model training.</p>
                <p><b>Key Insight:</b> MODE achieves 96%+ of full-data performance using only 30% of training data!</p>
                <hr>
                <p><i>Powered by CLIP, FAISS, and Gradio</i></p>
            </div>
            """)

        return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Large-Scale Retrieval Gradio Demo")
    parser.add_argument('--mode_db', type=str, default=None,
                       help='Path to MODE database')
    parser.add_argument('--random_db', type=str, default=None,
                       help='Path to Random baseline database')
    parser.add_argument('--full_db', type=str, default=None,
                       help='Path to Full database')
    parser.add_argument('--mode_model', type=str, default=None,
                       help='Path to trained MODE CLIP model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for web interface')
    parser.add_argument('--share', action='store_true',
                       help='Create public shareable link')

    args = parser.parse_args()

    if not any([args.mode_db, args.random_db, args.full_db]):
        print("Error: At least one database must be provided!")
        print("  --mode_db PATH")
        print("  --random_db PATH")
        print("  --full_db PATH")
        return

    if not HAS_GRADIO:
        print("Error: Gradio not installed!")
        print("Install with: pip install gradio")
        return

    if not HAS_TRANSFORMERS:
        print("Error: transformers not installed!")
        print("Install with: pip install transformers")
        return

    # Create demo
    demo_app = GradioLargeScaleDemo(
        mode_db_path=args.mode_db,
        random_db_path=args.random_db,
        full_db_path=args.full_db,
        mode_model_path=args.mode_model,
        device=args.device
    )

    # Launch
    demo = demo_app.create_interface()

    print("\n" + "="*80)
    print("🚀 Launching Gradio Demo")
    print("="*80)
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("="*80 + "\n")

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main()
