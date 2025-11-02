# 🔍 Interactive Retrieval Demo

Interactive tool for exploring Text-to-Image (T2I) and Image-to-Text (I2T) retrieval with your trained MODE CLIP model!

## 🎯 Features

- **Text-to-Image**: Enter text, find matching images
- **Image-to-Text**: Upload image, find matching captions
- **Web Interface**: Beautiful Gradio UI (recommended)
- **CLI Interface**: Command-line for quick experiments
- **Visualizations**: Pretty plots with similarity scores
- **Fast**: Pre-computed embeddings for instant retrieval

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install transformers torch gradio matplotlib pillow
```

### Step 2: Build Retrieval Database

```bash
# From your dataset
python build_retrieval_database.py \
    --data_dir ./datacomp_data \
    --model_path ./mode_output/final_clip_model.pt \
    --output_path ./retrieval_database.pt \
    --max_samples 1000
```

**What this does:**
- Loads your trained CLIP model
- Encodes all images and captions
- Saves pre-computed embeddings
- Takes ~5 minutes for 1000 samples

### Step 3: Launch Interactive Demo

**Option A: Web Interface (Recommended)**

```bash
python interactive_retrieval_demo.py \
    --interface web \
    --database_path ./retrieval_database.pt \
    --model_path ./mode_output/final_clip_model.pt
```

Then open your browser: `http://localhost:7860`

**Option B: Command Line Interface**

```bash
python interactive_retrieval_demo.py \
    --interface cli \
    --database_path ./retrieval_database.pt \
    --model_path ./mode_output/final_clip_model.pt
```

---

## 📸 Screenshots

### Web Interface

```
┌─────────────────────────────────────────────────────────┐
│  🔍 Interactive CLIP Retrieval Demo                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Tab: 📝 Text → Image                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Text Query:                                        │  │
│  │ "a cat sitting on a windowsill"                   │  │
│  │                                                     │  │
│  │ Number of Results: [5]                             │  │
│  │                                                     │  │
│  │ [🔍 Search]                                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  Results:                                                │
│  ┌─────┬─────┬─────┬─────┬─────┐                       │
│  │ #1  │ #2  │ #3  │ #4  │ #5  │                       │
│  │ 0.89│ 0.82│ 0.78│ 0.75│ 0.71│                       │
│  └─────┴─────┴─────┴─────┴─────┘                       │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Command Line Interface

```bash
> t2i a beautiful sunset over mountains

Searching for images matching: 'a beautiful sunset over mountains'
----------------------------------------------------------------------

Top 5 Results:
  1. Image #0042 - Score: 0.8734
     Caption: a stunning sunset with orange and pink clouds over...
  2. Image #0156 - Score: 0.8521
     Caption: mountain landscape at sunset with dramatic sky...
  3. Image #0289 - Score: 0.8312
     Caption: beautiful evening scene with sun setting behind peaks...
  4. Image #0337 - Score: 0.8145
     Caption: panoramic view of mountains during golden hour...
  5. Image #0421 - Score: 0.7998
     Caption: scenic mountain vista with colorful sunset...

Visualize results? (y/n): y
```

---

## 📖 Usage Examples

### Text-to-Image (T2I)

**Web Interface:**
1. Click "📝 Text → Image" tab
2. Enter text: `"a dog playing in a park"`
3. Adjust number of results (1-10)
4. Click "🔍 Search"
5. See results with similarity scores!

**Command Line:**
```bash
> t2i a dog playing in a park
> t2i beautiful sunset over mountains
> t2i people playing soccer
```

### Image-to-Text (I2T)

**Web Interface:**
1. Click "🖼️ Image → Text" tab
2. Upload an image (drag & drop or click)
3. Adjust number of results (1-10)
4. Click "🔍 Search"
5. See top matching captions!

**Command Line:**
```bash
> i2t /path/to/your/image.jpg
> i2t ./test_images/cat.png
```

---

## 🔧 Advanced Usage

### Build Database from MODE Selected Indices

```bash
# Use only MODE-selected samples (more diverse!)
python build_retrieval_database.py \
    --data_dir ./datacomp_data \
    --selected_indices ./datacomp_mode_cache/selected_indices.pt \
    --model_path ./mode_output/final_clip_model.pt \
    --output_path ./retrieval_database_mode.pt
```

### Use Different CLIP Model

```bash
# Try different CLIP variants
python interactive_retrieval_demo.py \
    --interface web \
    --database_path ./retrieval_database.pt \
    --clip_model_name openai/clip-vit-large-patch14
```

### Change Port (Web Interface)

```bash
python interactive_retrieval_demo.py \
    --interface web \
    --database_path ./retrieval_database.pt \
    --port 8080
```

### CPU-Only Mode

```bash
python interactive_retrieval_demo.py \
    --interface web \
    --database_path ./retrieval_database.pt \
    --device cpu
```

---

## 📊 Understanding the Results

### Similarity Scores

**Score Range:** 0.0 to 1.0 (higher = better match)

```
Score > 0.80:  Excellent match! 🟢
Score > 0.65:  Good match       🟡
Score > 0.50:  Fair match       🟠
Score < 0.50:  Weak match       🔴
```

### What Do Scores Mean?

- **0.90+**: Nearly perfect semantic match
- **0.80-0.90**: Strong alignment, likely correct
- **0.70-0.80**: Good match, minor differences
- **0.60-0.70**: Reasonable match, noticeable differences
- **<0.60**: Weak match, possibly wrong

### Example Interpretations

**Text-to-Image: "a cat sitting on a windowsill"**
```
Image #1 [0.89]: Cat on windowsill ✓ Perfect!
Image #2 [0.82]: Cat sitting indoors ✓ Good
Image #3 [0.75]: Cat near window    ✓ Reasonable
Image #4 [0.68]: Cat on furniture   ✓ Related
Image #5 [0.54]: Dog on windowsill  ✗ Wrong animal
```

**Image-to-Text: [Photo of pizza]**
```
Caption #1 [0.87]: "a delicious pizza with many toppings" ✓
Caption #2 [0.81]: "food on a plate with cheese" ✓
Caption #3 [0.73]: "italian cuisine on a table" ✓
Caption #4 [0.65]: "dinner with various dishes" ~
Caption #5 [0.52]: "a sandwich on a plate" ✗
```

---

## 🎨 Visualization Outputs

### Text-to-Image Visualization

Shows top-K retrieved images side-by-side with scores:

```
┌──────────────────────────────────────────────────────┐
│   Text-to-Image: "a dog playing in a park"          │
├──────┬──────┬──────┬──────┬──────┐
│      │      │      │      │      │
│ [1]  │ [2]  │ [3]  │ [4]  │ [5]  │
│      │      │      │      │      │
│ 0.85 │ 0.81 │ 0.78 │ 0.74 │ 0.69 │
└──────┴──────┴──────┴──────┴──────┘
```

### Image-to-Text Visualization

Shows query image + ranked captions with color-coded scores:

```
┌─────────────────┬────────────────────────────────┐
│                 │  Top Retrieved Captions        │
│   Query Image   │                                │
│                 │  1. [0.87] a dog playing...  🟢│
│    [Photo]      │  2. [0.82] a puppy with...   🟢│
│                 │  3. [0.76] an animal in...   🟡│
│                 │  4. [0.71] a pet outdoor...  🟡│
│                 │  5. [0.64] a dog sitting...  🟠│
└─────────────────┴────────────────────────────────┘
```

---

## 🧪 Demo Ideas

### 1. Test CLIP's Understanding

```bash
# Animals
t2i a photo of a cat
t2i a photo of a dog
t2i a photo of a bird

# Actions
t2i people running
t2i people sitting
t2i people dancing

# Attributes
t2i a red car
t2i a blue car
t2i a vintage car

# Scenes
t2i a beach at sunset
t2i a forest in winter
t2i a city at night
```

### 2. Test MODE vs Random

Build two databases and compare:

```bash
# MODE-selected data
python build_retrieval_database.py \
    --selected_indices ./datacomp_mode_cache/selected_indices.pt \
    --output_path ./db_mode.pt

# Random data
python build_retrieval_database.py \
    --data_dir ./datacomp_data \
    --output_path ./db_random.pt

# Compare results!
```

### 3. Cross-Modal Reasoning

Test if model understands relationships:

```bash
# Synonyms
t2i a happy person
t2i a joyful person

# Related concepts
t2i a doctor
t2i a hospital

# Negations (harder!)
t2i a day scene
t2i a night scene
```

---

## 🐛 Troubleshooting

### Issue: "No database loaded"

**Solution:**
```bash
# Make sure you built the database first
python build_retrieval_database.py \
    --data_dir ./datacomp_data \
    --output_path ./retrieval_database.pt

# Then provide it to demo
python interactive_retrieval_demo.py \
    --database_path ./retrieval_database.pt
```

### Issue: "gradio not installed"

**Solution:**
```bash
pip install gradio

# Or use CLI interface
python interactive_retrieval_demo.py --interface cli
```

### Issue: "Image not found"

**Solution:**
```bash
# For I2T, provide full path
i2t /full/path/to/image.jpg

# Or use relative path
i2t ./test_images/cat.png
```

### Issue: Slow retrieval

**Solution:**
- Pre-compute embeddings (use `build_retrieval_database.py`)
- Reduce database size (`--max_samples 500`)
- Use CPU if GPU memory limited (`--device cpu`)

### Issue: Web interface not loading

**Solution:**
```bash
# Try different port
python interactive_retrieval_demo.py --port 8080

# Check firewall settings
# Make sure port is not blocked
```

---

## 📈 Performance Tips

### Fast Retrieval
- **Pre-compute embeddings**: Build database once, reuse many times
- **Use GPU**: 10-100× faster than CPU
- **Batch encoding**: Processes multiple items together

### Large Databases
- **Index with FAISS**: For millions of samples
  ```python
  import faiss
  index = faiss.IndexFlatIP(embedding_dim)
  index.add(embeddings.numpy())
  ```
- **Approximate search**: Trade accuracy for speed
- **Quantization**: Reduce memory usage

### Quality Improvements
- **Fine-tune on your domain**: Better domain-specific retrieval
- **Ensemble models**: Combine multiple CLIP models
- **Re-ranking**: Use MODE scores for second-stage ranking

---

## 🎓 Use Cases

### 1. Research & Analysis
- Analyze what CLIP learned
- Compare MODE vs baseline models
- Test generalization to new domains
- Study failure cases

### 2. Data Exploration
- Find similar images quickly
- Discover dataset biases
- Identify mislabeled samples
- Curate subsets by query

### 3. Demo & Presentations
- Show MODE's capabilities
- Interactive paper supplement
- Live demo for reviewers
- Teaching tool for CLIP/VLMs

### 4. Application Development
- Prototype retrieval system
- Test query variations
- Benchmark performance
- User experience research

---

## 🚀 Extending the Demo

### Add Custom Filters

```python
# In interactive_retrieval_demo.py

def text_to_image_filtered(query_text, category='all', top_k=5):
    indices, scores = engine.text_to_image(query_text, top_k=100)

    # Filter by category
    filtered = [(i, s) for i, s in zip(indices, scores)
                if category == 'all' or get_category(i) == category]

    return filtered[:top_k]
```

### Add Multi-Modal Search

```python
def hybrid_search(text_query, image_query, text_weight=0.5, top_k=5):
    # Encode queries
    text_emb = encode_text(text_query)
    image_emb = encode_image(image_query)

    # Weighted combination
    query_emb = text_weight * text_emb + (1 - text_weight) * image_emb
    query_emb = F.normalize(query_emb)

    # Search
    similarities = query_emb @ image_embeddings.T
    return torch.topk(similarities, top_k)
```

### Add Feedback Loop

```python
def search_with_feedback(query, positive_ids, negative_ids):
    # Original query
    query_emb = encode_text(query)

    # Adjust based on feedback
    for pos_id in positive_ids:
        query_emb += 0.1 * image_embeddings[pos_id]

    for neg_id in negative_ids:
        query_emb -= 0.1 * image_embeddings[neg_id]

    query_emb = F.normalize(query_emb)

    # Re-search
    return search(query_emb)
```

---

## 📚 Related Tools

- **CLIP Interrogator**: Generate text from images
- **LAION Aesthetics**: Score image quality
- **OpenCLIP**: More CLIP variants
- **FAISS**: Fast similarity search at scale

---

## 📄 Citation

If you use this demo for your research:

```bibtex
@software{interactive_retrieval_demo,
  title={Interactive CLIP Retrieval Demo},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

---

## ✨ Tips for Best Results

1. **Use descriptive queries**: "a golden retriever playing fetch in a park" > "dog"
2. **Be specific**: "sunset over ocean" > "pretty picture"
3. **Try variations**: "happy person" vs "smiling person" vs "joyful person"
4. **Use MODE-selected data**: More diverse, better coverage
5. **Fine-tune for your domain**: Better domain-specific performance

---

## 🎉 Have Fun!

This tool is designed to help you:
- ✅ Explore your model's capabilities
- ✅ Find interesting examples for papers
- ✅ Debug retrieval issues
- ✅ Impress reviewers and collaborators

**Enjoy exploring! 🚀**

Questions? Check the code or open an issue.
