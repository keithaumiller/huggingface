#!/usr/bin/env python3
"""
Utility functions for Hugging Face model operations.
"""

import torch
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_names": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["device_names"].append(torch.cuda.get_device_name(i))
    
    return info


def estimate_model_size(model_name: str) -> Dict[str, Any]:
    """Estimate model size and requirements without loading the full model."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        # Rough estimation based on common architectures
        if hasattr(config, 'n_parameters'):
            params = config.n_parameters
        elif hasattr(config, 'num_parameters'):
            params = config.num_parameters
        else:
            # Estimate from config
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                # Rough estimate for transformer models
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, 'vocab_size', 50000)
                
                # Very rough estimation
                params = (hidden_size * hidden_size * 4 * num_layers) + (vocab_size * hidden_size * 2)
            else:
                params = "Unknown"
        
        if isinstance(params, (int, float)):
            size_mb = params * 4 / (1024 * 1024)  # Assuming float32
            size_gb = size_mb / 1024
            
            return {
                "model_name": model_name,
                "estimated_parameters": params,
                "estimated_size_mb": round(size_mb, 2),
                "estimated_size_gb": round(size_gb, 2),
                "config": config.to_dict()
            }
        else:
            return {
                "model_name": model_name,
                "estimated_parameters": "Unknown",
                "config": config.to_dict()
            }
    
    except Exception as e:
        return {
            "model_name": model_name,
            "error": str(e)
        }


def benchmark_model(model_name: str, num_runs: int = 5) -> Dict[str, Any]:
    """Benchmark model loading and inference time."""
    import time
    
    times = {
        "loading_times": [],
        "inference_times": [],
        "tokenization_times": []
    }
    
    test_text = "This is a test sentence for benchmarking the model performance."
    
    for run in range(num_runs):
        # Benchmark loading
        start_time = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            loading_time = time.time() - start_time
            times["loading_times"].append(loading_time)
            
            # Benchmark tokenization
            start_time = time.time()
            tokens = tokenizer(test_text, return_tensors="pt")
            tokenization_time = time.time() - start_time
            times["tokenization_times"].append(tokenization_time)
            
            # Benchmark inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**tokens)
            inference_time = time.time() - start_time
            times["inference_times"].append(inference_time)
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            return {"error": f"Run {run + 1}: {str(e)}"}
    
    # Calculate statistics
    stats = {}
    for key, values in times.items():
        if values:
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values
            }
    
    return {
        "model_name": model_name,
        "num_runs": num_runs,
        "benchmark_results": stats
    }


def create_model_comparison_chart(model_results: List[Dict[str, Any]], save_path: str = None):
    """Create a comparison chart for multiple models."""
    if not model_results:
        print("No model results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = []
    loading_times = []
    inference_times = []
    model_sizes = []
    parameters = []
    
    for result in model_results:
        if "error" in result:
            continue
            
        models.append(result.get("model_name", "Unknown"))
        
        # Extract benchmark data if available
        if "benchmark_results" in result:
            loading_times.append(result["benchmark_results"].get("loading_times", {}).get("mean", 0))
            inference_times.append(result["benchmark_results"].get("inference_times", {}).get("mean", 0))
        else:
            loading_times.append(0)
            inference_times.append(0)
        
        # Extract size data if available
        if "estimated_size_mb" in result:
            model_sizes.append(result["estimated_size_mb"])
        else:
            model_sizes.append(0)
            
        if "estimated_parameters" in result and isinstance(result["estimated_parameters"], (int, float)):
            parameters.append(result["estimated_parameters"] / 1e6)  # Convert to millions
        else:
            parameters.append(0)
    
    # Plot loading times
    if loading_times and any(t > 0 for t in loading_times):
        ax1.bar(models, loading_times)
        ax1.set_title("Model Loading Times")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot inference times
    if inference_times and any(t > 0 for t in inference_times):
        ax2.bar(models, inference_times)
        ax2.set_title("Inference Times")
        ax2.set_ylabel("Time (seconds)")
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot model sizes
    if model_sizes and any(s > 0 for s in model_sizes):
        ax3.bar(models, model_sizes)
        ax3.set_title("Model Sizes")
        ax3.set_ylabel("Size (MB)")
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot parameter counts
    if parameters and any(p > 0 for p in parameters):
        ax4.bar(models, parameters)
        ax4.set_title("Parameter Count")
        ax4.set_ylabel("Parameters (Millions)")
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    plt.show()


def analyze_tokenizer(model_name: str) -> Dict[str, Any]:
    """Analyze tokenizer properties."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test texts
        test_texts = [
            "Hello world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming our world.",
            "🤗 Hugging Face is awesome! 🚀"
        ]
        
        analysis = {
            "model_name": model_name,
            "tokenizer_type": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": getattr(tokenizer, 'model_max_length', 'Unknown'),
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "unk_token": tokenizer.unk_token,
                "bos_token": getattr(tokenizer, 'bos_token', None),
                "eos_token": getattr(tokenizer, 'eos_token', None),
                "cls_token": getattr(tokenizer, 'cls_token', None),
                "sep_token": getattr(tokenizer, 'sep_token', None),
                "mask_token": getattr(tokenizer, 'mask_token', None)
            },
            "tokenization_examples": []
        }
        
        # Analyze test texts
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            
            analysis["tokenization_examples"].append({
                "text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "num_tokens": len(tokens),
                "num_token_ids": len(token_ids)
            })
        
        return analysis
    
    except Exception as e:
        return {
            "model_name": model_name,
            "error": str(e)
        }


def save_results(results: Dict[str, Any], filename: str):
    """Save results to a JSON file."""
    output_dir = "/workspaces/huggingface/results"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    filepath = os.path.join("/workspaces/huggingface/results", filename)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def create_dataset_from_texts(texts: List[str], labels: Optional[List[Any]] = None) -> Dataset:
    """Create a Hugging Face dataset from texts and optional labels."""
    data = {"text": texts}
    
    if labels is not None:
        if len(labels) != len(texts):
            raise ValueError("Number of labels must match number of texts")
        data["label"] = labels
    
    return Dataset.from_dict(data)


def print_model_summary(model_name: str):
    """Print a comprehensive summary of a model."""
    print(f"🔍 Model Analysis: {model_name}")
    print("=" * 60)
    
    # Device info
    device_info = get_device_info()
    print(f"💻 Device Info:")
    print(f"   CUDA Available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"   GPU Count: {device_info['device_count']}")
        for i, name in enumerate(device_info['device_names']):
            print(f"   GPU {i}: {name}")
    print()
    
    # Model size estimation
    print("📏 Size Estimation:")
    size_info = estimate_model_size(model_name)
    if "error" not in size_info:
        if "estimated_parameters" in size_info:
            print(f"   Parameters: {size_info['estimated_parameters']:,}")
        if "estimated_size_mb" in size_info:
            print(f"   Size: {size_info['estimated_size_mb']} MB ({size_info['estimated_size_gb']} GB)")
    else:
        print(f"   Error: {size_info['error']}")
    print()
    
    # Tokenizer analysis
    print("🔤 Tokenizer Analysis:")
    tokenizer_info = analyze_tokenizer(model_name)
    if "error" not in tokenizer_info:
        print(f"   Type: {tokenizer_info['tokenizer_type']}")
        print(f"   Vocab Size: {tokenizer_info['vocab_size']:,}")
        print(f"   Max Length: {tokenizer_info['model_max_length']}")
        
        special_tokens = tokenizer_info['special_tokens']
        print("   Special Tokens:")
        for token_type, token in special_tokens.items():
            if token:
                print(f"     {token_type}: '{token}'")
    else:
        print(f"   Error: {tokenizer_info['error']}")


if __name__ == "__main__":
    # Example usage
    models_to_analyze = ["gpt2", "distilgpt2"]
    
    print("Hugging Face Model Utilities")
    print("============================")
    
    for model_name in models_to_analyze:
        print_model_summary(model_name)
        print("\n" + "-" * 60 + "\n")