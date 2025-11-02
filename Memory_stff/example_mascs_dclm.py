#!/usr/bin/env python3
"""
Example usage of MASCS-DCLM integration for language model data curation.
This script demonstrates various use cases and configurations.
"""

import json
import tempfile
import os
from pathlib import Path
import numpy as np
from mascs_dclm_integration import MASCSDCLMIntegrated, DCLMConfig, load_dataset_from_files

def create_sample_dataset(num_docs=1000, output_path="sample_dataset.jsonl"):
    """Create a sample dataset for demonstration"""

    # Sample topics and content templates
    topics = {
        "science": [
            "The scientific method involves systematic observation and experimentation.",
            "Climate change research shows significant warming trends over the past century.",
            "Quantum mechanics revolutionized our understanding of atomic behavior.",
            "Machine learning algorithms can identify patterns in complex datasets.",
            "Genetic engineering offers promising solutions for medical treatments."
        ],
        "technology": [
            "Artificial intelligence is transforming industries across the globe.",
            "Cloud computing provides scalable infrastructure for modern applications.",
            "Blockchain technology enables decentralized digital transactions.",
            "Internet of Things devices collect vast amounts of sensor data.",
            "Cybersecurity measures protect against evolving digital threats."
        ],
        "literature": [
            "The novel explores themes of identity and belonging in modern society.",
            "Poetry captures the essence of human emotion through carefully chosen words.",
            "Character development drives the narrative forward in compelling ways.",
            "Literary analysis reveals deeper meanings within seemingly simple texts.",
            "The author's use of symbolism enhances the story's impact."
        ],
        "history": [
            "Ancient civilizations developed sophisticated systems of governance and trade.",
            "Industrial revolution brought massive changes to society and economy.",
            "World wars reshaped international relations and political boundaries.",
            "Cultural exchange along trade routes influenced artistic and religious practices.",
            "Archaeological discoveries provide insights into prehistoric human behavior."
        ]
    }

    documents = []

    for i in range(num_docs):
        # Randomly select topic and base content
        topic = np.random.choice(list(topics.keys()))
        base_content = np.random.choice(topics[topic])

        # Add some variation to content length and quality
        if np.random.random() < 0.7:  # 70% high quality
            # Expand content for higher quality documents
            expanded_content = base_content
            for _ in range(np.random.randint(2, 5)):
                expanded_content += " " + np.random.choice(topics[topic])

            text = expanded_content
            quality_tier = "high"
        elif np.random.random() < 0.5:  # Some medium quality
            text = base_content + " " + np.random.choice(topics[topic])
            quality_tier = "medium"
        else:  # Some low quality (short, repetitive, or noisy)
            if np.random.random() < 0.5:
                text = base_content[:50]  # Too short
            else:
                text = base_content + " " + base_content  # Repetitive
            quality_tier = "low"

        # Add some noise to simulate real-world data
        if np.random.random() < 0.1:
            text = text + " [NOISE] random characters 123 !@#"

        doc = {
            "id": f"doc_{i:04d}",
            "text": text,
            "topic": topic,
            "quality_tier": quality_tier,
            "source": "synthetic",
            "length": len(text),
            "metadata": {
                "created": f"2024-01-{np.random.randint(1, 29):02d}",
                "synthetic": True
            }
        }
        documents.append(doc)

    # Save to JSONL
    with open(output_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    print(f"Created sample dataset with {len(documents)} documents at {output_path}")
    return documents

def example_basic_usage():
    """Demonstrate basic usage of MASCS-DCLM integration"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)

    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        sample_path = f.name

    create_sample_dataset(num_docs=500, output_path=sample_path)

    # Configure DCLM
    dclm_config = DCLMConfig(
        data_source=sample_path,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        max_seq_length=1024,
        min_text_length=20,
        quality_threshold=0.3,
        cache_dir="./example_cache"
    )

    # Configure MASCS
    mascs_config = {
        'budget': 100,  # Select top 100 documents
        'memory_window': 50,
        'device': 'cpu'  # Use CPU for example
    }

    # Initialize selector
    selector = MASCSDCLMIntegrated(dclm_config, mascs_config)

    # Load and process documents
    raw_documents = load_dataset_from_files(sample_path)
    processed_documents, selected_indices = selector.process_and_select(
        raw_documents,
        target_topics=["science", "technology"]
    )

    # Print results
    print(f"Original documents: {len(raw_documents)}")
    print(f"After filtering: {len(processed_documents)}")
    print(f"Selected: {len(selected_indices)}")
    print(f"Selection ratio: {len(selected_indices)/len(raw_documents):.2%}")

    # Analyze selected documents
    selected_docs = [processed_documents[i] for i in selected_indices]
    avg_quality = np.mean([doc.get('quality_score', 0) for doc in selected_docs])
    print(f"Average quality score: {avg_quality:.3f}")

    # Show strategy weights
    print("\nFinal strategy weights:")
    for strategy, weight in selector.current_weights.items():
        print(f"  {strategy}: {weight:.3f}")

    # Clean up
    os.unlink(sample_path)

def example_advanced_configuration():
    """Demonstrate advanced configuration and features"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Configuration")
    print("="*60)

    # Create more diverse sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        sample_path = f.name

    create_sample_dataset(num_docs=1000, output_path=sample_path)

    # Advanced DCLM configuration
    dclm_config = DCLMConfig(
        data_source=sample_path,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        max_seq_length=2048,
        min_text_length=50,
        max_text_length=5000,
        deduplication_threshold=0.9,
        quality_threshold=0.5,  # Higher quality threshold
        cache_dir="./advanced_cache"
    )

    # Advanced MASCS configuration
    mascs_config = {
        'budget': 200,
        'memory_window': 100,
        'device': 'cpu'
    }

    # Initialize with custom settings
    selector = MASCSDCLMIntegrated(dclm_config, mascs_config)

    # Load documents
    raw_documents = load_dataset_from_files(sample_path)

    # Process with multiple target topics
    processed_documents, selected_indices = selector.process_and_select(
        raw_documents,
        target_topics=["artificial intelligence", "machine learning", "quantum mechanics", "climate change"]
    )

    print(f"Advanced filtering results:")
    print(f"  Original: {len(raw_documents)}")
    print(f"  Filtered: {len(processed_documents)}")
    print(f"  Selected: {len(selected_indices)}")

    # Analyze quality distribution
    selected_docs = [processed_documents[i] for i in selected_indices]
    quality_scores = [doc.get('quality_score', 0) for doc in selected_docs]

    print(f"\nQuality analysis:")
    print(f"  Mean quality: {np.mean(quality_scores):.3f}")
    print(f"  Std quality: {np.std(quality_scores):.3f}")
    print(f"  Min quality: {np.min(quality_scores):.3f}")
    print(f"  Max quality: {np.max(quality_scores):.3f}")

    # Analyze topic distribution
    topic_counts = {}
    for doc in selected_docs:
        original_doc = raw_documents[processed_documents.index(doc)]
        topic = original_doc.get('topic', 'unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print(f"\nTopic distribution in selected documents:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count} ({count/len(selected_docs)*100:.1f}%)")

    # Save state for future use
    state_path = "./selector_state_advanced.pkl"
    selector.save_state(state_path)
    print(f"\nSelector state saved to {state_path}")

    # Clean up
    os.unlink(sample_path)

def example_iterative_selection():
    """Demonstrate iterative selection with performance feedback"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Iterative Selection with Performance Feedback")
    print("="*60)

    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        sample_path = f.name

    create_sample_dataset(num_docs=800, output_path=sample_path)

    # Configuration
    dclm_config = DCLMConfig(
        data_source=sample_path,
        quality_threshold=0.4,
        cache_dir="./iterative_cache"
    )

    mascs_config = {
        'budget': 100,
        'memory_window': 75,
        'device': 'cpu'
    }

    selector = MASCSDCLMIntegrated(dclm_config, mascs_config)
    raw_documents = load_dataset_from_files(sample_path)

    # Simulate iterative selection with performance feedback
    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Select documents
        processed_documents, selected_indices = selector.process_and_select(
            raw_documents,
            target_topics=["science", "technology"]
        )

        selected_docs = [processed_documents[i] for i in selected_indices]

        # Simulate performance evaluation (in real use, this would be actual model performance)
        simulated_performance = np.random.uniform(0.7, 0.9)  # Random performance between 70-90%
        performance_delta = simulated_performance - 0.8  # Compare to baseline of 80%

        print(f"Selected {len(selected_docs)} documents")
        print(f"Simulated performance: {simulated_performance:.3f}")
        print(f"Performance delta: {performance_delta:+.3f}")

        # Update strategy weights based on performance
        selector.update_strategy_weights(performance_delta)

        print("Updated strategy weights:")
        for strategy, weight in selector.current_weights.items():
            print(f"  {strategy}: {weight:.3f}")

        # Store performance for next iteration
        selector.performance_history.append(simulated_performance)

    print(f"\nPerformance history: {selector.performance_history}")

    # Clean up
    os.unlink(sample_path)

def example_custom_extension():
    """Demonstrate how to extend the system with custom components"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Extension")
    print("="*60)

    class CustomMASCSDCLM(MASCSDCLMIntegrated):
        """Extended version with custom scoring strategy"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add custom strategy
            self.strategy_names.append('S_CUSTOM')
            self.current_weights['S_CUSTOM'] = 1.0 / len(self.strategy_names)
            # Renormalize weights
            total = sum(self.current_weights.values())
            for k in self.current_weights:
                self.current_weights[k] /= total

        def compute_custom_score(self, document: dict) -> float:
            """Custom scoring based on word diversity"""
            text = document.get('text', '')
            words = text.lower().split()

            if len(words) == 0:
                return 0.0

            unique_words = len(set(words))
            diversity_score = unique_words / len(words)

            return diversity_score

        def select_coreset_documents(self, documents, **kwargs):
            """Override to include custom scoring"""
            # Compute custom scores
            custom_scores = []
            for doc in documents:
                custom_score = self.compute_custom_score(doc)
                custom_scores.append(custom_score)

            # Store custom scores
            if not hasattr(self, '_custom_scores_cache'):
                self._custom_scores_cache = {}

            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f'doc_{i}')
                self._custom_scores_cache[doc_id] = custom_scores[i]

            # Call parent method
            return super().select_coreset_documents(documents, **kwargs)

    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        sample_path = f.name

    create_sample_dataset(num_docs=300, output_path=sample_path)

    # Use custom selector
    dclm_config = DCLMConfig(
        data_source=sample_path,
        quality_threshold=0.3,
        cache_dir="./custom_cache"
    )

    mascs_config = {
        'budget': 50,
        'memory_window': 30,
        'device': 'cpu'
    }

    custom_selector = CustomMASCSDCLM(dclm_config, mascs_config)
    raw_documents = load_dataset_from_files(sample_path)

    # Process with custom selector
    processed_documents, selected_indices = custom_selector.process_and_select(
        raw_documents
    )

    print(f"Custom selection results:")
    print(f"  Original: {len(raw_documents)}")
    print(f"  Selected: {len(selected_indices)}")

    print(f"\nCustom strategy weights (including S_CUSTOM):")
    for strategy, weight in custom_selector.current_weights.items():
        print(f"  {strategy}: {weight:.3f}")

    # Analyze custom scores
    if hasattr(custom_selector, '_custom_scores_cache'):
        custom_scores = list(custom_selector._custom_scores_cache.values())
        print(f"\nCustom score statistics:")
        print(f"  Mean: {np.mean(custom_scores):.3f}")
        print(f"  Std: {np.std(custom_scores):.3f}")

    # Clean up
    os.unlink(sample_path)

def main():
    """Run all examples"""
    print("MASCS-DCLM Integration Examples")
    print("="*60)

    # Create cache directories
    for cache_dir in ["./example_cache", "./advanced_cache", "./iterative_cache", "./custom_cache"]:
        os.makedirs(cache_dir, exist_ok=True)

    try:
        # Run examples
        example_basic_usage()
        example_advanced_configuration()
        example_iterative_selection()
        example_custom_extension()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up cache directories
        import shutil
        for cache_dir in ["./example_cache", "./advanced_cache", "./iterative_cache", "./custom_cache"]:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

        # Clean up any remaining files
        for file_pattern in ["selector_state_advanced.pkl"]:
            if os.path.exists(file_pattern):
                os.unlink(file_pattern)

if __name__ == "__main__":
    main()