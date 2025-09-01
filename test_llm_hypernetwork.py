#!/usr/bin/env python3
"""
Test script for LLM hypernetwork integration
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import List

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_imports():
    """Test all LLM hypernetwork imports"""
    print("Testing LLM imports...")
    try:
        from llm_hypernetworks import (
            LLMMultiScoringHypernetwork,
            LLMCoresetSelector,
            LLMTrainingState,
            create_llm_hypernetwork,
            PerplexityScoring,
            TokenDiversityScoring,
            AttentionCoverageScoring,
            GradientAlignmentScoring,
            TokenImportanceScoring,
            RepetitionPenaltyScoring
        )
        from llm_experiments import (
            LLMGaLoreSelector,
            TextDataset,
            run_llm_experiment
        )
        from config import LLMConfig, LLM_QUICK_CONFIG
        print("✓ All LLM imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_llm_scoring_functions():
    """Test LLM-specific scoring functions"""
    print("\nTesting LLM scoring functions...")
    
    try:
        from llm_hypernetworks import (
            PerplexityScoring,
            TokenDiversityScoring,
            create_llm_scoring_functions
        )
        
        # Create a simple mock model
        class MockLLMModel(nn.Module):
            def __init__(self, vocab_size=50257, hidden_size=768):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
                    num_layers=2
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids=None, attention_mask=None, labels=None, 
                       output_hidden_states=False, output_attentions=False, **kwargs):
                # Simple forward pass
                embeds = self.embeddings(input_ids)
                hidden = self.transformer(embeds)
                logits = self.lm_head(hidden)
                
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                class Output:
                    pass
                
                output = Output()
                output.loss = loss if loss is not None else torch.tensor(1.0)
                output.logits = logits
                
                if output_hidden_states:
                    output.hidden_states = [hidden]
                if output_attentions:
                    # Mock attention weights
                    seq_len = input_ids.size(1)
                    output.attentions = [torch.rand(1, 8, seq_len, seq_len)]
                
                return output
            
            def get_input_embeddings(self):
                return self.embeddings
        
        # Create mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50257
                self.pad_token = '[PAD]'
                self.eos_token = '[EOS]'
                
            def __call__(self, text, **kwargs):
                # Return mock tokenized output
                seq_len = kwargs.get('max_length', 128)
                return {
                    'input_ids': torch.randint(0, self.vocab_size, (1, seq_len)),
                    'attention_mask': torch.ones(1, seq_len)
                }
            
            def encode(self, text, **kwargs):
                return list(range(min(len(text), 512)))
        
        model = MockLLMModel()
        tokenizer = MockTokenizer()
        device = torch.device('cpu')
        
        # Test creating scoring functions
        scoring_functions = create_llm_scoring_functions(model, tokenizer, device)
        print(f"✓ Created {len(scoring_functions)} LLM scoring functions")
        
        # Test each scoring function
        text_data = "This is a test sentence for LLM scoring."
        selected_set = []
        context = {}
        
        for sf in scoring_functions[:3]:  # Test first 3 for speed
            score = sf.score(0, text_data, selected_set, context)
            print(f"  {sf.name}: {score:.4f}")
        
        print("✓ LLM scoring functions working")
        return True
        
    except Exception as e:
        print(f"✗ Scoring function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_hypernetwork():
    """Test LLM hypernetwork architecture"""
    print("\nTesting LLM hypernetwork...")
    
    try:
        from llm_hypernetworks import (
            LLMMultiScoringHypernetwork,
            LLMTrainingState,
            LLMScoringFunction
        )
        
        # Create mock scoring functions
        class MockScoring(LLMScoringFunction):
            def __init__(self, name):
                super().__init__(name, None, None, 'cpu')
                
            def score(self, idx, text_data, selected_set, context):
                return np.random.random()
        
        scoring_functions = [MockScoring(f"mock_{i}") for i in range(6)]
        
        # Create hypernetwork
        hypernetwork = LLMMultiScoringHypernetwork(
            scoring_functions=scoring_functions,
            state_dim=15,
            hidden_dim=64,
            num_heads=2,
            dropout=0.1
        )
        
        print(f"✓ Created LLM hypernetwork with {len(scoring_functions)} functions")
        
        # Test forward pass
        test_state = LLMTrainingState(
            epoch=1,
            loss=2.5,
            perplexity=12.0,
            gradient_norm=1.5,
            learning_rate=0.001,
            tokens_seen=10000,
            total_tokens=100000,
            avg_sequence_length=256,
            vocab_coverage=0.3,
            performance_history=[15.0, 14.0, 13.0, 12.5, 12.0],
            attention_entropy=0.7
        )
        
        with torch.no_grad():
            weights, temperature, value = hypernetwork(test_state)
        
        print(f"✓ Forward pass successful")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Temperature: {temperature.item():.3f}")
        print(f"  Value: {value.item():.3f}")
        print(f"  Weight distribution: {weights.tolist()}")
        
        # Check weight sum
        assert abs(weights.sum().item() - 1.0) < 0.1, "Weights should approximately sum to 1"
        print("✓ Weight normalization verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Hypernetwork test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_selector():
    """Test LLM coreset selector"""
    print("\nTesting LLM coreset selector...")
    
    try:
        from llm_experiments import LLMGaLoreSelector, TextDataset
        from llm_hypernetworks import create_llm_hypernetwork
        
        # Create mock model and tokenizer
        from test_llm_hypernetwork import MockLLMModel, MockTokenizer
        model = MockLLMModel()
        tokenizer = MockTokenizer()
        
        # Create mock datasets
        train_texts = ["Sample text " * 10 for _ in range(100)]
        val_texts = ["Validation text " * 10 for _ in range(20)]
        
        train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
        val_dataset = TextDataset(val_texts, tokenizer, max_length=128)
        
        # Initialize selector
        selector = LLMGaLoreSelector(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            memory_budget_mb=100,
            rank=64,
            use_llm_hypernetwork=True
        )
        
        print("✓ LLM selector initialized")
        
        # Test coreset selection
        selected_indices, selection_info = selector.select_coreset_llm(
            budget=10,
            current_performance=15.0
        )
        
        print(f"✓ Selected {len(selected_indices)} samples")
        print(f"  Strategy: {selection_info.get('strategy', 'N/A')}")
        print(f"  Perplexity: {selection_info.get('perplexity', 'N/A'):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_config():
    """Test LLM configuration"""
    print("\nTesting LLM configuration...")
    
    try:
        from config import LLMConfig, LLM_QUICK_CONFIG, get_config
        
        # Test LLM config creation
        config = LLMConfig()
        assert hasattr(config, 'use_llm_hypernetwork')
        assert hasattr(config, 'llm_hypernet_hidden_dim')
        assert hasattr(config, 'model_name')
        print("✓ LLM config has all required attributes")
        
        # Test quick config
        quick_config = LLM_QUICK_CONFIG
        print(f"✓ Quick config: epochs={quick_config.epochs}, "
              f"samples={quick_config.max_train_samples}")
        
        # Test get_config
        llm_config = get_config('llm_quick')
        print(f"✓ Retrieved LLM config: {llm_config.model_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_mini_training():
    """Test a minimal training loop"""
    print("\nTesting mini training loop...")
    
    try:
        from llm_experiments import train_llm_epoch, TextDataset
        from torch.utils.data import DataLoader
        import torch.optim as optim
        
        # Create mock components
        from test_llm_hypernetwork import MockLLMModel, MockTokenizer
        model = MockLLMModel()
        tokenizer = MockTokenizer()
        
        # Create small dataset
        texts = ["Test sentence " * 5 for _ in range(20)]
        dataset = TextDataset(texts, tokenizer, max_length=64)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Run one training step
        device = torch.device('cpu')
        metrics = train_llm_epoch(
            model, dataloader, optimizer, scheduler, device,
            gradient_accumulation_steps=2
        )
        
        print(f"✓ Training step completed")
        print(f"  Loss: {metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add mock classes to module for import
class MockLLMModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, 
               output_hidden_states=False, output_attentions=False, **kwargs):
        embeds = self.embeddings(input_ids)
        hidden = self.transformer(embeds)
        logits = self.lm_head(hidden)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        class Output:
            pass
        
        output = Output()
        output.loss = loss if loss is not None else torch.tensor(1.0)
        output.logits = logits
        
        if output_hidden_states:
            output.hidden_states = [hidden]
        if output_attentions:
            seq_len = input_ids.size(1)
            output.attentions = [torch.rand(1, 8, seq_len, seq_len)]
        
        return output
    
    def get_input_embeddings(self):
        return self.embeddings

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 50257
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
        
    def __call__(self, text, **kwargs):
        seq_len = kwargs.get('max_length', 128)
        return {
            'input_ids': torch.randint(0, self.vocab_size, (1, seq_len)),
            'attention_mask': torch.ones(1, seq_len)
        }
    
    def encode(self, text, **kwargs):
        return list(range(min(len(text), 512)))

if __name__ == "__main__":
    print("LLM HYPERNETWORK INTEGRATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_llm_imports()
    all_passed &= test_llm_scoring_functions()
    all_passed &= test_llm_hypernetwork()
    all_passed &= test_llm_selector()
    all_passed &= test_llm_config()
    all_passed &= test_mini_training()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nLLM hypernetwork integration is working correctly.")
        print("\nTo run LLM experiments:")
        print("  python llm_experiments.py --model gpt2 --epochs 5 --use_hypernetwork")
        print("\nOptimizations for faster convergence:")
        print("  - 6 specialized LLM scoring functions")
        print("  - Attention-aware diversity metrics")
        print("  - Perplexity-based selection")
        print("  - Token importance scoring")
        print("  - Gradient accumulation for large models")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)