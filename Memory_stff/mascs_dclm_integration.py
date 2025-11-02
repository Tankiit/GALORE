import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from dataclasses import dataclass
import ray
from torch.utils.data import Dataset, DataLoader
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoTokenizer, AutoModel
import datasets
from datasets import Dataset as HFDataset
import faiss
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DCLMConfig:
    """Configuration for DCLM-style data processing"""
    data_source: str
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    min_text_length: int = 10
    max_text_length: int = 100000
    deduplication_threshold: float = 0.8
    quality_threshold: float = 0.5
    cache_dir: str = "./dclm_cache"
    use_ray: bool = True
    ray_num_cpus: int = 4

class DCLMDataProcessor:
    """DCLM-style data processor with filtering capabilities"""

    def __init__(self, config: DCLMConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Initialize deduplication index
        self.dedup_db_path = self.cache_dir / "deduplication.db"
        self._init_dedup_db()

        # Initialize quality scorer
        self.quality_model = None
        self._init_quality_scorer()

    def _init_dedup_db(self):
        """Initialize SQLite database for deduplication"""
        conn = sqlite3.connect(self.dedup_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS document_hashes (
                hash TEXT PRIMARY KEY,
                source TEXT,
                timestamp INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _init_quality_scorer(self):
        """Initialize quality scoring model"""
        try:
            # Use a lightweight model for quality scoring
            from sentence_transformers import SentenceTransformer
            self.quality_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence_transformers not available, using simple quality scoring")
            self.quality_model = None

    def compute_text_hash(self, text: str) -> str:
        """Compute hash for deduplication"""
        # Normalize text for better deduplication
        normalized = ' '.join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate"""
        text_hash = self.compute_text_hash(text)
        conn = sqlite3.connect(self.dedup_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT hash FROM document_hashes WHERE hash = ?', (text_hash,))
        is_dup = cursor.fetchone() is not None

        if not is_dup:
            cursor.execute('INSERT INTO document_hashes VALUES (?, ?, ?)',
                         (text_hash, 'unknown', int(time.time())))
            conn.commit()

        conn.close()
        return is_dup

    def compute_quality_score(self, text: str) -> float:
        """Compute quality score for text"""
        if self.quality_model is None:
            # Simple heuristic-based quality scoring
            score = 0.0

            # Length-based scoring
            if self.config.min_text_length <= len(text) <= self.config.max_text_length:
                score += 0.3

            # Basic language quality checks
            words = text.split()
            if len(words) > 5:
                score += 0.2

            # Check for reasonable sentence structure
            sentences = text.split('.')
            if len(sentences) > 1:
                score += 0.2

            # Check for excessive repetition
            unique_words = len(set(words))
            if len(words) > 0 and unique_words / len(words) > 0.5:
                score += 0.3

            return score
        else:
            # Use sentence transformer for quality scoring
            try:
                embedding = self.quality_model.encode([text])
                # Simple quality metric based on embedding norm
                return min(1.0, np.linalg.norm(embedding[0]) / 10.0)
            except:
                return 0.5

    def filter_document(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply DCLM-style filtering to a document"""
        text = document.get('text', '')

        # Basic length filtering
        if len(text) < self.config.min_text_length or len(text) > self.config.max_text_length:
            return None

        # Deduplication
        if self.is_duplicate(text):
            return None

        # Quality filtering
        quality_score = self.compute_quality_score(text)
        if quality_score < self.config.quality_threshold:
            return None

        # Add metadata
        document['quality_score'] = quality_score
        document['processed_timestamp'] = time.time()

        return document

    def tokenize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize document text"""
        text = document.get('text', '')

        # Tokenize with truncation
        tokens = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        document['input_ids'] = tokens['input_ids'].squeeze().tolist()
        document['attention_mask'] = tokens['attention_mask'].squeeze().tolist()
        document['token_count'] = len(document['input_ids'])

        return document

    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents"""
        processed = []
        for doc in documents:
            filtered_doc = self.filter_document(doc)
            if filtered_doc is not None:
                tokenized_doc = self.tokenize_document(filtered_doc)
                processed.append(tokenized_doc)
        return processed

class MASCSDCLMIntegrated:
    """Enhanced MASCS with DCLM integration for language model data selection"""

    def __init__(self, dclm_config: DCLMConfig, mascs_config: Dict[str, Any]):
        self.dclm_config = dclm_config
        self.mascs_config = mascs_config

        # Initialize DCLM processor
        self.dclm_processor = DCLMDataProcessor(dclm_config)

        # Initialize MASCS components
        self.memory_window = mascs_config.get('memory_window', 100)
        self.device = mascs_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.budget = mascs_config.get('budget', 10000)

        # Memory structures for documents
        self.document_memories = {}
        self.memory_buffer = deque(maxlen=self.memory_window * 100)
        self.selection_history = defaultdict(int)
        self.quality_history = defaultdict(list)

        # Strategy components
        self.strategy_names = ['S_Q', 'S_D', 'S_L', 'S_R', 'S_C', 'S_T']  # Quality, Diversity, Length, Rarity, Coherence, Topic
        self.current_weights = {name: 1.0/len(self.strategy_names) for name in self.strategy_names}

        # Feature extractors
        self.text_embedder = None
        self._init_text_embedder()

        # Caching
        self._cached_embeddings = {}
        self._cached_scores = {}

        # Performance tracking
        self.performance_history = []

    def _init_text_embedder(self):
        """Initialize text embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence_transformers not available, using simple embeddings")
            self.text_embedder = None

    def compute_document_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for document text"""
        if text in self._cached_embeddings:
            return self._cached_embeddings[text]

        if self.text_embedder is not None:
            embedding = self.text_embedder.encode([text])[0]
        else:
            # Simple bag-of-words embedding as fallback
            words = text.lower().split()
            vocab_size = 1000  # Simple vocabulary
            embedding = np.zeros(vocab_size)
            for word in words:
                word_hash = hash(word) % vocab_size
                embedding[word_hash] += 1
            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)

        self._cached_embeddings[text] = embedding
        return embedding

    def compute_quality_score(self, document: Dict[str, Any]) -> float:
        """Compute quality score using DCLM processor"""
        text = document.get('text', '')
        if text in self._cached_scores.get('quality', {}):
            return self._cached_scores['quality'][text]

        score = self.dclm_processor.compute_quality_score(text)

        if 'quality' not in self._cached_scores:
            self._cached_scores['quality'] = {}
        self._cached_scores['quality'][text] = score

        return score

    def compute_diversity_score(self, document: Dict[str, Any], selected_documents: List[Dict[str, Any]]) -> float:
        """Compute diversity score based on embedding similarity"""
        if len(selected_documents) == 0:
            return 1.0

        doc_embedding = self.compute_document_embedding(document.get('text', ''))

        # Compute similarity to already selected documents
        similarities = []
        for selected_doc in selected_documents:
            selected_embedding = self.compute_document_embedding(selected_doc.get('text', ''))
            similarity = np.dot(doc_embedding, selected_embedding)
            similarities.append(similarity)

        # Diversity is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        return 1.0 - max_similarity

    def compute_length_score(self, document: Dict[str, Any]) -> float:
        """Compute length-based score"""
        text = document.get('text', '')
        token_count = document.get('token_count', len(text.split()))

        # Prefer documents with moderate length
        optimal_length = self.dclm_config.max_seq_length // 2
        length_diff = abs(token_count - optimal_length)
        return max(0.0, 1.0 - (length_diff / optimal_length))

    def compute_rarity_score(self, document: Dict[str, Any], corpus_stats: Dict[str, Any]) -> float:
        """Compute rarity score based on n-gram frequencies"""
        text = document.get('text', '')
        words = text.lower().split()

        if len(words) == 0:
            return 0.0

        # Compute n-gram rarity
        word_frequencies = corpus_stats.get('word_frequencies', {})
        total_words = corpus_stats.get('total_words', 1)

        rare_word_count = 0
        for word in words:
            word_freq = word_frequencies.get(word, 1)
            if word_freq / total_words < 0.001:  # Rare words threshold
                rare_word_count += 1

        return rare_word_count / len(words)

    def compute_coherence_score(self, document: Dict[str, Any]) -> float:
        """Compute coherence score based on text structure"""
        text = document.get('text', '')
        sentences = text.split('.')

        if len(sentences) < 2:
            return 0.5

        # Simple coherence metric based on sentence similarity
        if self.text_embedder is not None:
            sentence_embeddings = [self.text_embedder.encode([sent.strip()])[0]
                                 for sent in sentences if sent.strip()]

            if len(sentence_embeddings) < 2:
                return 0.5

            # Average pairwise similarity between consecutive sentences
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = np.dot(sentence_embeddings[i], sentence_embeddings[i+1])
                similarities.append(sim)

            return np.mean(similarities) if similarities else 0.5
        else:
            # Simple heuristic: consistent sentence length
            sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
            if len(sentence_lengths) == 0:
                return 0.5
            length_std = np.std(sentence_lengths)
            length_mean = np.mean(sentence_lengths)
            coefficient_of_variation = length_std / length_mean if length_mean > 0 else 1
            return max(0.0, 1.0 - coefficient_of_variation)

    def compute_topic_score(self, document: Dict[str, Any], target_topics: List[str] = None) -> float:
        """Compute topic relevance score"""
        if target_topics is None:
            return 0.5

        text = document.get('text', '').lower()

        # Simple keyword-based topic scoring
        topic_matches = 0
        for topic in target_topics:
            topic_keywords = topic.lower().split()
            for keyword in topic_keywords:
                if keyword in text:
                    topic_matches += 1

        return min(1.0, topic_matches / len(target_topics))

    def update_memory(self, doc_id: str, scores: Dict[str, float], selected: bool, performance_delta: float = 0.0):
        """Update memory for document selection history"""
        memory_vector = np.array([
            scores.get('S_Q', 0), scores.get('S_D', 0), scores.get('S_L', 0),
            scores.get('S_R', 0), scores.get('S_C', 0), scores.get('S_T', 0),
            float(selected), performance_delta
        ], dtype=np.float32)

        if doc_id not in self.document_memories:
            self.document_memories[doc_id] = deque(maxlen=self.memory_window)

        self.document_memories[doc_id].append(memory_vector)
        self.memory_buffer.append((doc_id, memory_vector))

        if selected:
            self.selection_history[doc_id] += 1
            if performance_delta != 0:
                self.quality_history[doc_id].append(performance_delta)

    def select_coreset_documents(self, documents: List[Dict[str, Any]],
                                corpus_stats: Dict[str, Any] = None,
                                target_topics: List[str] = None,
                                selected_documents: List[Dict[str, Any]] = None) -> List[int]:
        """Select coreset of documents using integrated MASCS-DCLM approach"""

        if corpus_stats is None:
            corpus_stats = self._compute_corpus_stats(documents)

        if selected_documents is None:
            selected_documents = []

        # Compute scores for all documents
        all_scores = {strategy: [] for strategy in self.strategy_names}
        doc_ids = []

        logger.info(f"Computing scores for {len(documents)} documents...")

        for i, doc in enumerate(tqdm(documents, desc="Computing scores")):
            doc_id = doc.get('id', f'doc_{i}')
            doc_ids.append(doc_id)

            # Compute individual strategy scores
            scores = {
                'S_Q': self.compute_quality_score(doc),
                'S_D': self.compute_diversity_score(doc, selected_documents),
                'S_L': self.compute_length_score(doc),
                'S_R': self.compute_rarity_score(doc, corpus_stats),
                'S_C': self.compute_coherence_score(doc),
                'S_T': self.compute_topic_score(doc, target_topics)
            }

            for strategy in self.strategy_names:
                all_scores[strategy].append(scores[strategy])

        # Normalize scores
        for strategy in self.strategy_names:
            scores_array = np.array(all_scores[strategy])
            if np.max(scores_array) > 0:
                all_scores[strategy] = (scores_array / np.max(scores_array)).tolist()

        # Combine scores using current strategy weights
        combined_scores = np.zeros(len(documents))
        for strategy in self.strategy_names:
            weight = self.current_weights[strategy]
            strategy_scores = np.array(all_scores[strategy])
            combined_scores += weight * strategy_scores

        # Select top-k documents
        selected_indices = np.argsort(combined_scores)[-self.budget:].tolist()

        # Update memory
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i]
            is_selected = i in selected_indices
            doc_scores = {strategy: all_scores[strategy][i] for strategy in self.strategy_names}
            self.update_memory(doc_id, doc_scores, is_selected)

        logger.info(f"Selected {len(selected_indices)} documents out of {len(documents)}")

        return selected_indices

    def _compute_corpus_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute corpus-level statistics for rarity scoring"""
        word_frequencies = defaultdict(int)
        total_words = 0

        for doc in documents:
            text = doc.get('text', '')
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                word_frequencies[word] += 1

        return {
            'word_frequencies': dict(word_frequencies),
            'total_words': total_words,
            'vocabulary_size': len(word_frequencies),
            'avg_doc_length': total_words / len(documents) if documents else 0
        }

    def process_and_select(self, raw_documents: List[Dict[str, Any]],
                          target_topics: List[str] = None) -> Tuple[List[Dict[str, Any]], List[int]]:
        """End-to-end processing: filter, tokenize, and select documents"""

        logger.info(f"Processing {len(raw_documents)} raw documents...")

        # Step 1: DCLM-style filtering and processing
        processed_documents = []
        for doc in tqdm(raw_documents, desc="Filtering documents"):
            processed_doc = self.dclm_processor.filter_document(doc)
            if processed_doc is not None:
                tokenized_doc = self.dclm_processor.tokenize_document(processed_doc)
                processed_documents.append(tokenized_doc)

        logger.info(f"After filtering: {len(processed_documents)} documents remain")

        if len(processed_documents) == 0:
            return [], []

        # Step 2: MASCS-based coreset selection
        selected_indices = self.select_coreset_documents(
            processed_documents,
            target_topics=target_topics
        )

        return processed_documents, selected_indices

    def update_strategy_weights(self, performance_delta: float):
        """Update strategy weights based on performance feedback"""
        # Simple adaptive weighting based on recent performance
        if performance_delta > 0:
            # Increase weights of strategies that led to good performance
            for strategy in self.strategy_names:
                recent_usage = sum(1 for _, memory in self.memory_buffer
                                 if len(memory) > 6 and memory[6] > 0.5)  # Selected documents
                if recent_usage > 0:
                    self.current_weights[strategy] *= 1.1
        else:
            # Decrease weights slightly for poor performance
            for strategy in self.strategy_names:
                self.current_weights[strategy] *= 0.95

        # Normalize weights
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_names:
                self.current_weights[strategy] /= total_weight

    def save_state(self, filepath: str):
        """Save the current state of the selector"""
        state = {
            'document_memories': dict(self.document_memories),
            'selection_history': dict(self.selection_history),
            'quality_history': dict(self.quality_history),
            'current_weights': self.current_weights,
            'performance_history': self.performance_history,
            'cached_embeddings': self._cached_embeddings,
            'cached_scores': self._cached_scores
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load the state of the selector"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.document_memories = defaultdict(lambda: deque(maxlen=self.memory_window),
                                           state['document_memories'])
        self.selection_history = defaultdict(int, state['selection_history'])
        self.quality_history = defaultdict(list, state['quality_history'])
        self.current_weights = state['current_weights']
        self.performance_history = state['performance_history']
        self._cached_embeddings = state.get('cached_embeddings', {})
        self._cached_scores = state.get('cached_scores', {})

        logger.info(f"State loaded from {filepath}")

def load_dataset_from_files(data_path: str, file_pattern: str = "*.jsonl") -> List[Dict[str, Any]]:
    """Load dataset from JSONL files"""
    documents = []
    data_path = Path(data_path)

    if data_path.is_file() and data_path.suffix == '.jsonl':
        files = [data_path]
    else:
        files = list(data_path.glob(file_pattern))

    for file_path in files:
        logger.info(f"Loading {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    doc['id'] = doc.get('id', f"{file_path.stem}_{line_num}")
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num + 1}: {e}")

    logger.info(f"Loaded {len(documents)} documents")
    return documents

def main():
    parser = argparse.ArgumentParser(description='MASCS-DCLM Integrated Document Selection')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset files (JSONL format)')
    parser.add_argument('--output_path', type=str, default='./selected_documents.jsonl',
                       help='Output path for selected documents')
    parser.add_argument('--budget', type=int, default=10000,
                       help='Number of documents to select')
    parser.add_argument('--cache_dir', type=str, default='./dclm_cache',
                       help='Cache directory for DCLM processing')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b',
                       help='Tokenizer name')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--quality_threshold', type=float, default=0.3,
                       help='Quality threshold for filtering')
    parser.add_argument('--target_topics', type=str, nargs='*',
                       help='Target topics for topic-based selection')
    parser.add_argument('--save_state', type=str,
                       help='Path to save selector state')
    parser.add_argument('--load_state', type=str,
                       help='Path to load selector state')

    args = parser.parse_args()

    # Configure DCLM processor
    dclm_config = DCLMConfig(
        data_source=args.data_path,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        quality_threshold=args.quality_threshold,
        cache_dir=args.cache_dir
    )

    # Configure MASCS
    mascs_config = {
        'budget': args.budget,
        'memory_window': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Initialize integrated selector
    logger.info("Initializing MASCS-DCLM integrated selector...")
    selector = MASCSDCLMIntegrated(dclm_config, mascs_config)

    # Load previous state if specified
    if args.load_state and os.path.exists(args.load_state):
        selector.load_state(args.load_state)

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    raw_documents = load_dataset_from_files(args.data_path)

    if not raw_documents:
        logger.error("No documents loaded. Check your data path and file format.")
        return

    # Process and select documents
    processed_documents, selected_indices = selector.process_and_select(
        raw_documents,
        target_topics=args.target_topics
    )

    # Save selected documents
    selected_documents = [processed_documents[i] for i in selected_indices]

    logger.info(f"Saving {len(selected_documents)} selected documents to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for doc in selected_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    # Save statistics
    stats_path = Path(args.output_path).with_suffix('.stats.json')
    stats = {
        'total_documents': len(raw_documents),
        'processed_documents': len(processed_documents),
        'selected_documents': len(selected_documents),
        'selection_ratio': len(selected_documents) / len(raw_documents) if raw_documents else 0,
        'strategy_weights': selector.current_weights,
        'average_quality_score': np.mean([doc.get('quality_score', 0) for doc in selected_documents]),
        'average_token_count': np.mean([doc.get('token_count', 0) for doc in selected_documents])
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Selection statistics saved to {stats_path}")

    # Save state if specified
    if args.save_state:
        selector.save_state(args.save_state)

    logger.info("Document selection completed successfully!")
    print(f"\nSelection Summary:")
    print(f"  Original documents: {len(raw_documents)}")
    print(f"  After filtering: {len(processed_documents)}")
    print(f"  Selected: {len(selected_documents)}")
    print(f"  Selection ratio: {stats['selection_ratio']:.2%}")
    print(f"  Average quality score: {stats['average_quality_score']:.3f}")
    print(f"  Average token count: {stats['average_token_count']:.0f}")

if __name__ == "__main__":
    main()