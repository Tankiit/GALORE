#!/usr/bin/env python3
"""
Simple Embedding-Based Retrieval Demo

Works with embedding databases - no images needed!
Shows MODE's value through text-based retrieval.
"""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import json

try:
    import gradio as gr
    import open_clip
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "gradio", "open-clip-torch"])
    import gradio as gr
    import open_clip


class EmbeddingRetrievalEngine:
    """Simple retrieval using embeddings."""

    def __init__(self, db_path: str, model_name: str = "ViT-B-32"):
        print(f"Loading database from {db_path}...")
        self.db = torch.load(db_path, map_location='cpu')

        self.num_samples = self.db['num_samples']
        self.captions = self.db['captions']
        self.text_embeddings = self.db['text_embeddings']
        self.image_embeddings = self.db['image_embeddings']

        print(f"✓ Loaded {self.num_samples} samples")
        print(f"  Embedding dim: {self.db['embedding_dim']}")

        # Load CLIP for encoding queries
        print(f"Loading CLIP model: {model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        print("✓ Ready for retrieval!")

    def text_query(self, query: str, top_k: int = 5):
        """Retrieve captions similar to query text."""
        # Encode query
        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            query_emb = self.model.encode_text(text_tokens)
            query_emb = F.normalize(query_emb, dim=-1).cpu()

        # Compute similarities
        similarities = (query_emb @ self.text_embeddings.T).squeeze()

        # Get top-k
        top_k = min(top_k, len(similarities))
        scores, indices = torch.topk(similarities, top_k)

        results = []
        for idx, score in zip(indices, scores):
            results.append({
                'caption': self.captions[idx],
                'score': float(score),
                'index': int(idx)
            })

        return results


def create_demo(db_paths: dict):
    """Create Gradio demo interface."""

    # Load retrieval engines
    engines = {}
    for name, path in db_paths.items():
        if Path(path).exists():
            engines[name] = EmbeddingRetrievalEngine(path)
        else:
            print(f"Warning: Database not found: {path}")

    if not engines:
        raise ValueError("No databases loaded!")

    def search(query: str, top_k: int, compare: bool):
        """Perform retrieval."""
        if not query.strip():
            return "Please enter a query"

        results_html = f"<h2>Query: \"{query}\"</h2>\n"

        # Compute statistics for comparison
        if compare and len(engines) > 1:
            stats = {}
            for name, engine in engines.items():
                results = engine.text_query(query, top_k)
                scores = [r['score'] for r in results]
                stats[name] = {
                    'avg': sum(scores) / len(scores),
                    'max': max(scores),
                    'min': min(scores)
                }

            # Show comparison table
            results_html += "\n<h3>📊 Comparison Statistics</h3>\n"
            results_html += "<table style='width:100%; border-collapse: collapse;'>\n"
            results_html += "<tr style='background: #f0f0f0;'>"
            results_html += "<th style='padding:8px; border:1px solid #ddd;'>Method</th>"
            results_html += "<th style='padding:8px; border:1px solid #ddd;'>Avg Score</th>"
            results_html += "<th style='padding:8px; border:1px solid #ddd;'>Max Score</th>"
            results_html += "<th style='padding:8px; border:1px solid #ddd;'>Min Score</th>"
            results_html += "</tr>\n"

            for name, s in stats.items():
                results_html += "<tr>"
                results_html += f"<td style='padding:8px; border:1px solid #ddd;'><b>{name}</b></td>"
                results_html += f"<td style='padding:8px; border:1px solid #ddd;'>{s['avg']:.3f}</td>"
                results_html += f"<td style='padding:8px; border:1px solid #ddd;'>{s['max']:.3f}</td>"
                results_html += f"<td style='padding:8px; border:1px solid #ddd;'>{s['min']:.3f}</td>"
                results_html += "</tr>\n"

            results_html += "</table>\n<br>\n"

        # Show results for each engine
        for name, engine in engines.items():
            results = engine.text_query(query, top_k)

            results_html += f"\n<h3>🔍 {name} Results</h3>\n"
            results_html += "<ol style='line-height: 2;'>\n"

            for r in results:
                score = r['score']
                # Color code by score
                if score >= 0.8:
                    color = "#4CAF50"  # Green
                elif score >= 0.6:
                    color = "#FF9800"  # Orange
                else:
                    color = "#F44336"  # Red

                results_html += f"<li>"
                results_html += f"<span style='background:{color}; color:white; padding:2px 6px; border-radius:3px; margin-right:8px;'>"
                results_html += f"{score:.3f}</span>"
                results_html += f"<b>{r['caption']}</b>"
                results_html += f"</li>\n"

            results_html += "</ol>\n"

        return results_html

    # Create interface
    with gr.Blocks(title="MODE Embedding Retrieval Demo") as demo:
        gr.Markdown("""
        # 🔍 MODE Embedding Retrieval Demo

        Demonstration of MODE's value using embedding-based retrieval.

        **How it works**: Enter a text query and see the most similar captions retrieved
        from MODE-selected data vs other methods.

        **Why this matters**: MODE selects diverse, informative samples that provide
        better retrieval coverage even with less data!
        """)

        # Show database info
        info_md = "### 📊 Loaded Databases\n"
        for name, engine in engines.items():
            info_md += f"- **{name}**: {engine.num_samples} samples "
            info_md += f"({engine.db['embedding_dim']} dim embeddings)\n"

        gr.Markdown(info_md)

        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="a dog playing in a park",
                    lines=2
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of results"
                    )

                    compare_checkbox = gr.Checkbox(
                        label="Show comparison statistics",
                        value=len(engines) > 1
                    )

                search_button = gr.Button("🔍 Search", variant="primary")

                # Example queries
                gr.Markdown("### 💡 Example Queries")
                examples = [
                    "a cat sitting on a windowsill",
                    "beautiful sunset over mountains",
                    "people playing soccer in a park",
                    "red car on a city street",
                    "close-up of a flower"
                ]

                with gr.Row():
                    for ex in examples[:3]:
                        gr.Button(ex, size="sm").click(
                            lambda x=ex: x, outputs=query_input
                        )

            with gr.Column():
                results_output = gr.HTML(label="Results")

        search_button.click(
            fn=search,
            inputs=[query_input, top_k_slider, compare_checkbox],
            outputs=results_output
        )

        gr.Markdown("""
        ---
        ### 📖 About This Demo

        This demo uses **embedding-based retrieval** - matching your query against
        pre-computed text embeddings.

        **Key Insight**: Even without images, MODE's selection strategy results in:
        - Better semantic coverage
        - More diverse retrieved results
        - Higher average similarity scores

        In production, you'd retrieve actual images. This demo proves MODE's value
        works at the embedding level!
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch embedding retrieval demo")
    parser.add_argument(
        "--mode_db",
        type=str,
        default="./demo_embedding_db/embedding_database.pt",
        help="Path to MODE database"
    )
    parser.add_argument(
        "--random_db",
        type=str,
        help="Path to Random database (optional)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )

    args = parser.parse_args()

    # Collect databases
    db_paths = {}

    if Path(args.mode_db).exists():
        db_paths["MODE"] = args.mode_db
    else:
        print(f"Error: MODE database not found: {args.mode_db}")
        print("\nBuild it first:")
        print("  python build_embedding_demo.py")
        return

    if args.random_db and Path(args.random_db).exists():
        db_paths["Random"] = args.random_db

    print("\n" + "="*80)
    print("LAUNCHING EMBEDDING RETRIEVAL DEMO")
    print("="*80)
    print(f"Databases: {list(db_paths.keys())}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("="*80 + "\n")

    # Create and launch demo
    demo = create_demo(db_paths)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
