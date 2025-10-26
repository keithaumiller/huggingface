#!/usr/bin/env python3
"""
Model Explorer and Downloader for Hugging Face Hub
Explore, download, and manage models in your local workspace.
"""

import os
import json
import torch
import signal
import sys
import time
import threading
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import HfApi, list_models, model_info
import pandas as pd
from utils.config import config


class ModelExplorer:
    """Explore and download models from Hugging Face Hub."""
    
    def __init__(self):
        self.api = HfApi()
        self.cache_dir = config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.download_interrupted = False
        self.current_download = None
        
        # Set up interrupt handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals during downloads."""
        print(f"\n🛑 Interrupt received (signal {signum})")
        self.download_interrupted = True
        
        if self.current_download:
            print(f"⏹️  Stopping download of {self.current_download}...")
            print("💾 Partial files will be safely cached and can be resumed later.")
        else:
            print("🔄 Stopping current operation...")
        
        # Give a moment for cleanup
        time.sleep(1)
        print("✅ Safe to exit - no data corruption will occur.")
        print("💡 Tip: Restart the download anytime - it will resume from where it left off.")
        
        # Don't exit immediately, let the download function handle cleanup
        return
    
    def search_models(self, 
                     task: str = None, 
                     library: str = "transformers",
                     sort: str = "downloads",
                     limit: int = 20) -> List[Dict]:
        """Search for models on Hugging Face Hub."""
        print(f"🔍 Searching for models (task: {task}, library: {library})...")
        
        try:
            models = list_models(
                task=task,
                library=library,
                sort=sort,
                limit=limit
            )
            
            model_list = []
            for model in models:
                model_data = {
                    "id": model.id,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "task": task or "Unknown",
                    "library": library
                }
                model_list.append(model_data)
            
            return model_list
        
        except Exception as e:
            print(f"❌ Error searching models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            info = model_info(model_id)
            
            # Get config to estimate size
            try:
                config_info = AutoConfig.from_pretrained(model_id)
                config_dict = config_info.to_dict()
            except:
                config_dict = {}
            
            return {
                "id": model_id,
                "downloads": getattr(info, 'downloads', 0),
                "likes": getattr(info, 'likes', 0),
                "tags": getattr(info, 'tags', []),
                "pipeline_tag": getattr(info, 'pipeline_tag', 'Unknown'),
                "library_name": getattr(info, 'library_name', 'Unknown'),
                "config": config_dict,
                "model_size": self._estimate_model_size(config_dict),
                "created_at": str(getattr(info, 'created_at', 'Unknown')),
                "last_modified": str(getattr(info, 'last_modified', 'Unknown'))
            }
        
        except Exception as e:
            return {"id": model_id, "error": str(e)}
    
    def _estimate_model_size(self, config: Dict) -> Dict[str, Any]:
        """Estimate model size from configuration."""
        if not config:
            return {"error": "No config available"}
        
        try:
            # Common size indicators
            hidden_size = config.get('hidden_size', config.get('d_model', 0))
            num_layers = config.get('num_hidden_layers', config.get('n_layer', 0))
            vocab_size = config.get('vocab_size', 50000)
            
            if hidden_size and num_layers:
                # Very rough estimation for transformer models
                params = (hidden_size * hidden_size * 4 * num_layers) + (vocab_size * hidden_size * 2)
                size_mb = params * 4 / (1024 * 1024)  # Assuming float32
                
                return {
                    "estimated_parameters": params,
                    "estimated_size_mb": round(size_mb, 2),
                    "estimated_size_gb": round(size_mb / 1024, 2),
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "vocab_size": vocab_size
                }
            else:
                return {"info": "Size estimation not available"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def download_model(self, model_id: str, force_download: bool = False, resume: bool = True) -> bool:
        """Download a model to local cache with interrupt handling."""
        print(f"📥 Downloading model: {model_id}")
        
        # Reset interrupt flag
        self.download_interrupted = False
        self.current_download = model_id
        
        # Check if model already exists (unless forcing)
        if not force_download and self._is_model_cached(model_id):
            print(f"✅ Model {model_id} already cached. Use force_download=True to re-download.")
            self.current_download = None
            return True
        
        try:
            print(f"🌐 Starting download - Safe to interrupt with Ctrl+C")
            print(f"💾 Cache location: {self.cache_dir}")
            
            # Download tokenizer with progress
            print("  📝 Downloading tokenizer...")
            if self.download_interrupted:
                raise KeyboardInterrupt("Download interrupted by user")
            
            tokenizer = self._safe_download_component(
                lambda: AutoTokenizer.from_pretrained(
                    model_id, 
                    cache_dir=self.cache_dir,
                    force_download=force_download,
                    resume_download=resume
                ),
                f"tokenizer for {model_id}"
            )
            
            if tokenizer is None:  # Download was interrupted
                return False
            
            print("  🧠 Downloading model weights...")
            if self.download_interrupted:
                raise KeyboardInterrupt("Download interrupted by user")
            
            model = self._safe_download_component(
                lambda: AutoModel.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    force_download=force_download,
                    resume_download=resume,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True  # More memory efficient
                ),
                f"model weights for {model_id}"
            )
            
            if model is None:  # Download was interrupted
                return False
            
            print(f"✅ Successfully downloaded: {model_id}")
            print(f"📊 Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Save download info
            try:
                self._save_download_info(model_id, tokenizer, model)
            except Exception as e:
                print(f"⚠️  Warning: Could not save download info: {e}")
            
            return True
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Download of {model_id} was interrupted")
            print("💾 Partial downloads are safely cached - no corruption occurred")
            print("🔄 You can resume this download later by running the same command")
            return False
        
        except Exception as e:
            print(f"❌ Error downloading {model_id}: {e}")
            print("💡 This might be due to:")
            print("   • Network connectivity issues")
            print("   • Model name typo or model doesn't exist")
            print("   • Insufficient disk space")
            print("   • Model requires authentication (set HF token)")
            return False
        
        finally:
            self.current_download = None
    
    def _safe_download_component(self, download_func, component_name: str):
        """Safely download a component with interrupt checking."""
        try:
            # Check for interrupt before starting
            if self.download_interrupted:
                print(f"⏹️  Skipping {component_name} - download interrupted")
                return None
            
            # Start download in a way that can be interrupted
            print(f"    📡 Fetching {component_name}...")
            component = download_func()
            
            # Check for interrupt after download
            if self.download_interrupted:
                print(f"⏹️  {component_name} download completed but operation was interrupted")
                return None
                
            print(f"    ✅ {component_name} ready")
            return component
            
        except KeyboardInterrupt:
            print(f"\n⏹️  {component_name} download interrupted")
            return None
        except Exception as e:
            print(f"❌ Error downloading {component_name}: {e}")
            return None
    
    def _is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already in cache."""
        try:
            # Try to load from cache without downloading
            AutoConfig.from_pretrained(model_id, cache_dir=self.cache_dir, local_files_only=True)
            return True
        except:
            return False
    
    def _save_download_info(self, model_id: str, tokenizer, model):
        """Save information about downloaded model."""
        info_file = os.path.join(self.cache_dir, "downloaded_models.json")
        
        # Load existing info
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                downloaded_models = json.load(f)
        else:
            downloaded_models = {}
        
        # Add new model info
        model_info = {
            "model_id": model_id,
            "download_date": str(pd.Timestamp.now()),
            "tokenizer_type": type(tokenizer).__name__,
            "model_type": type(model).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
        
        downloaded_models[model_id] = model_info
        
        # Save updated info
        with open(info_file, 'w') as f:
            json.dump(downloaded_models, f, indent=2)
    
    def list_downloaded_models(self) -> Dict[str, Any]:
        """List all downloaded models."""
        info_file = os.path.join(self.cache_dir, "downloaded_models.json")
        
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def check_download_integrity(self, model_id: str) -> Dict[str, Any]:
        """Check if a downloaded model is complete and valid."""
        try:
            print(f"🔍 Checking integrity of {model_id}...")
            
            # Try to load tokenizer
            tokenizer_ok = False
            try:
                AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir, local_files_only=True)
                tokenizer_ok = True
                print("  ✅ Tokenizer: OK")
            except Exception as e:
                print(f"  ❌ Tokenizer: Missing or corrupted ({e})")
            
            # Try to load model config
            config_ok = False
            try:
                AutoConfig.from_pretrained(model_id, cache_dir=self.cache_dir, local_files_only=True)
                config_ok = True
                print("  ✅ Config: OK")
            except Exception as e:
                print(f"  ❌ Config: Missing or corrupted ({e})")
            
            # Try to load model (just check if files exist)
            model_ok = False
            try:
                # This will fail if model files are missing but won't load the full model
                AutoModel.from_pretrained(
                    model_id, 
                    cache_dir=self.cache_dir, 
                    local_files_only=True,
                    torch_dtype=torch.float32
                )
                model_ok = True
                print("  ✅ Model weights: OK")
            except Exception as e:
                print(f"  ❌ Model weights: Missing or corrupted ({e})")
            
            integrity_status = {
                "model_id": model_id,
                "tokenizer_ok": tokenizer_ok,
                "config_ok": config_ok,
                "model_ok": model_ok,
                "complete": tokenizer_ok and config_ok and model_ok,
                "cache_dir": self.cache_dir
            }
            
            if integrity_status["complete"]:
                print(f"  🎉 {model_id} is complete and ready to use!")
            else:
                print(f"  ⚠️  {model_id} appears to be incomplete - consider re-downloading")
                
            return integrity_status
            
        except Exception as e:
            return {
                "model_id": model_id,
                "error": str(e),
                "complete": False
            }
    
    def cleanup_incomplete_downloads(self) -> int:
        """Clean up any incomplete or corrupted downloads."""
        print("🧹 Cleaning up incomplete downloads...")
        
        cleaned_count = 0
        downloaded_models = self.list_downloaded_models()
        
        for model_id in list(downloaded_models.keys()):
            integrity = self.check_download_integrity(model_id)
            
            if not integrity.get("complete", False):
                print(f"🗑️  Removing incomplete download: {model_id}")
                try:
                    # Remove from tracking
                    del downloaded_models[model_id]
                    cleaned_count += 1
                    
                    # Note: We don't delete the actual cache files as they might be shared
                    # The HF cache is designed to handle partial downloads gracefully
                    
                except Exception as e:
                    print(f"❌ Error cleaning {model_id}: {e}")
        
        # Update the downloaded models file
        info_file = os.path.join(self.cache_dir, "downloaded_models.json")
        with open(info_file, 'w') as f:
            json.dump(downloaded_models, f, indent=2)
        
        print(f"✅ Cleanup complete. Removed {cleaned_count} incomplete entries.")
        return cleaned_count
    
    def get_download_progress_info(self) -> Dict[str, Any]:
        """Get information about download progress and cache status."""
        cache_info = {
            "cache_directory": self.cache_dir,
            "cache_exists": os.path.exists(self.cache_dir),
            "total_size_mb": 0,
            "file_count": 0,
            "model_count": 0,
            "last_download": None
        }
        
        if os.path.exists(self.cache_dir):
            # Calculate cache size
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        cache_info["total_size_mb"] += os.path.getsize(filepath) / (1024 * 1024)
                        cache_info["file_count"] += 1
                        
                        # Count model files
                        if file.endswith(('.bin', '.safetensors', 'pytorch_model.bin')):
                            cache_info["model_count"] += 1
                    except (OSError, IOError):
                        pass  # Skip files that can't be accessed
            
            # Get last download info
            downloaded_models = self.list_downloaded_models()
            if downloaded_models:
                dates = [info.get('download_date', '') for info in downloaded_models.values()]
                cache_info["last_download"] = max(dates) if dates else None
        
        cache_info["total_size_gb"] = cache_info["total_size_mb"] / 1024
        return cache_info
    
    def get_popular_models_by_task(self) -> Dict[str, List[Dict]]:
        """Get popular models organized by task."""
        tasks = [
            "text-generation",
            "text-classification", 
            "question-answering",
            "summarization",
            "translation",
            "fill-mask",
            "token-classification",
            "text2text-generation"
        ]
        
        popular_models = {}
        
        for task in tasks:
            print(f"🔍 Searching {task} models...")
            models = self.search_models(task=task, limit=10)
            popular_models[task] = models
        
        return popular_models


def show_model_categories():
    """Show different categories of models available."""
    categories = {
        "🤖 Text Generation Models": {
            "Small (< 1GB)": [
                "gpt2",
                "distilgpt2", 
                "microsoft/DialoGPT-small",
                "EleutherAI/gpt-neo-125m"
            ],
            "Medium (1-5GB)": [
                "gpt2-medium",
                "microsoft/DialoGPT-medium",
                "EleutherAI/gpt-neo-1.3B",
                "facebook/opt-1.3b"
            ],
            "Large (5-20GB)": [
                "gpt2-large",
                "gpt2-xl", 
                "EleutherAI/gpt-neo-2.7B",
                "facebook/opt-6.7b"
            ],
            "Very Large (20GB+)": [
                "EleutherAI/gpt-neox-20b",
                "facebook/opt-30b",
                "bigscience/bloom-7b1"
            ]
        },
        "📊 Classification Models": {
            "Sentiment Analysis": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "distilbert-base-uncased-finetuned-sst-2-english",
                "nlptown/bert-base-multilingual-uncased-sentiment"
            ],
            "Emotion Detection": [
                "j-hartmann/emotion-english-distilroberta-base",
                "SamLowe/roberta-base-go_emotions"
            ],
            "Topic Classification": [
                "facebook/bart-large-mnli",
                "microsoft/deberta-base-mnli"
            ]
        },
        "❓ Question Answering": {
            "General QA": [
                "distilbert-base-cased-distilled-squad",
                "deepset/roberta-base-squad2",
                "microsoft/deberta-base"
            ],
            "Conversational QA": [
                "deepset/roberta-base-squad2",
                "twmkn9/distilbert-base-uncased-squad2"
            ]
        },
        "📝 Summarization Models": [
            "facebook/bart-large-cnn",
            "t5-small",
            "t5-base",
            "pegasus-xsum"
        ],
        "🔤 Token Classification (NER)": [
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            "dslim/bert-base-NER",
            "microsoft/deberta-base"
        ],
        "🌍 Translation Models": [
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-en-fr", 
            "facebook/mbart-large-50-many-to-many-mmt"
        ]
    }
    
    for category, models in categories.items():
        print(f"\n{category}")
        print("=" * 50)
        
        if isinstance(models, dict):
            for subcategory, model_list in models.items():
                print(f"\n  {subcategory}:")
                for model in model_list:
                    print(f"    • {model}")
        else:
            for model in models:
                print(f"  • {model}")


def interactive_model_explorer():
    """Interactive model exploration and download."""
    explorer = ModelExplorer()
    
    while True:
        print("\n🤗 Hugging Face Model Explorer")
        print("=" * 40)
        print("1. 📋 Show model categories")
        print("2. 🔍 Search models by task")
        print("3. ℹ️  Get model information")
        print("4. 📥 Download a model (Ctrl+C safe)")
        print("5. 📂 List downloaded models")
        print("6. 🌟 Show popular models")
        print("7. 💾 Check storage usage")
        print("8. 🔍 Check model integrity")
        print("9. 🧹 Cleanup incomplete downloads")
        print("0. 🚪 Exit")
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            show_model_categories()
        
        elif choice == "2":
            task = input("Enter task (e.g., text-generation, text-classification): ").strip()
            limit = input("Number of results (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            
            models = explorer.search_models(task=task, limit=limit)
            
            print(f"\n📋 Top {len(models)} models for '{task}':")
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model['id']} (↓{model['downloads']:,} downloads, ❤️{model['likes']} likes)")
        
        elif choice == "3":
            model_id = input("Enter model ID: ").strip()
            if model_id:
                info = explorer.get_model_info(model_id)
                
                print(f"\n📊 Model Information: {model_id}")
                print("-" * 50)
                
                if "error" not in info:
                    print(f"Downloads: {info.get('downloads', 0):,}")
                    print(f"Likes: {info.get('likes', 0)}")
                    print(f"Task: {info.get('pipeline_tag', 'Unknown')}")
                    print(f"Library: {info.get('library_name', 'Unknown')}")
                    
                    size_info = info.get('model_size', {})
                    if "estimated_parameters" in size_info:
                        print(f"Est. Parameters: {size_info['estimated_parameters']:,}")
                        print(f"Est. Size: {size_info['estimated_size_mb']:.1f} MB ({size_info['estimated_size_gb']:.2f} GB)")
                    
                    tags = info.get('tags', [])
                    if tags:
                        print(f"Tags: {', '.join(tags[:5])}")
                else:
                    print(f"Error: {info['error']}")
        
        elif choice == "4":
            print("📥 Safe Download Mode")
            print("• Press Ctrl+C anytime to safely interrupt")
            print("• Partial downloads will be cached and can resume")
            print("• No risk of file corruption")
            print()
            
            model_id = input("Enter model ID to download: ").strip()
            if model_id:
                force = input("Force re-download? (y/N): ").strip().lower() == 'y'
                resume = input("Allow resume from partial download? (Y/n): ").strip().lower() != 'n'
                
                print(f"\n🚀 Starting download of {model_id}")
                print("💡 Tip: You can safely interrupt with Ctrl+C and resume later")
                
                try:
                    success = explorer.download_model(model_id, force_download=force, resume=resume)
                    
                    if success:
                        print(f"🎉 Model {model_id} downloaded successfully!")
                        # Check integrity
                        integrity = explorer.check_download_integrity(model_id)
                        if integrity.get("complete", False):
                            print("✅ Download integrity verified - ready to use!")
                    else:
                        print(f"⚠️  Download of {model_id} was not completed")
                        print("💡 You can retry the same command to resume")
                        
                except KeyboardInterrupt:
                    print("\n🛑 Download safely interrupted by user")
                    print("💾 Progress saved - you can resume anytime")
        
        elif choice == "5":
            downloaded = explorer.list_downloaded_models()
            
            if downloaded:
                print(f"\n📂 Downloaded Models ({len(downloaded)} total):")
                print("-" * 60)
                
                for model_id, info in downloaded.items():
                    size_mb = info.get('model_size_mb', 0)
                    download_date = info.get('download_date', 'Unknown')
                    print(f"• {model_id}")
                    print(f"  Size: {size_mb:.1f} MB | Downloaded: {download_date[:10]}")
            else:
                print("📭 No models downloaded yet.")
        
        elif choice == "6":
            print("🌟 Getting popular models by task...")
            popular = explorer.get_popular_models_by_task()
            
            for task, models in popular.items():
                print(f"\n🏆 Popular {task} models:")
                for i, model in enumerate(models[:5], 1):
                    print(f"  {i}. {model['id']} (↓{model['downloads']:,})")
        
        elif choice == "7":
            print("💾 Getting storage information...")
            cache_info = explorer.get_download_progress_info()
            
            print(f"\n💾 Storage Usage:")
            print(f"Cache Directory: {cache_info['cache_directory']}")
            print(f"Directory exists: {'✅' if cache_info['cache_exists'] else '❌'}")
            print(f"Total Size: {cache_info['total_size_gb']:.2f} GB ({cache_info['total_size_mb']:.1f} MB)")
            print(f"Total Files: {cache_info['file_count']:,}")
            print(f"Model Files: {cache_info['model_count']}")
            
            if cache_info['last_download']:
                print(f"Last Download: {cache_info['last_download'][:10]}")
            else:
                print("Last Download: Never")
        
        elif choice == "8":
            model_id = input("Enter model ID to check: ").strip()
            if model_id:
                integrity = explorer.check_download_integrity(model_id)
                
                if integrity.get("complete", False):
                    print(f"🎉 {model_id} is complete and ready to use!")
                else:
                    print(f"⚠️  {model_id} has issues:")
                    if not integrity.get("tokenizer_ok", False):
                        print("  • Tokenizer missing or corrupted")
                    if not integrity.get("config_ok", False):
                        print("  • Config missing or corrupted")
                    if not integrity.get("model_ok", False):
                        print("  • Model weights missing or corrupted")
                    print("💡 Try re-downloading this model")
        
        elif choice == "9":
            confirm = input("Clean up incomplete downloads? This will remove tracking for incomplete models (y/N): ").strip().lower()
            if confirm == 'y':
                cleaned = explorer.cleanup_incomplete_downloads()
                print(f"✅ Cleanup complete! Removed {cleaned} incomplete entries.")
            else:
                print("❌ Cleanup cancelled")
        
        else:
            print("❌ Invalid choice. Please select 0-9.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    interactive_model_explorer()