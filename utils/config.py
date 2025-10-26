#!/usr/bin/env python3
"""
Configuration manager for the Hugging Face workspace.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration management class."""
    
    def __init__(self, config_file: str = "/workspaces/huggingface/config/config.json"):
        # Load environment variables from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        
        # Load configuration from JSON file
        self.config_file = config_file
        self.config = self._load_config()
        
        # Override with environment variables where available
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            return {}
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Hugging Face settings
        if os.getenv("HUGGINGFACE_HUB_TOKEN"):
            self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Model settings
        if os.getenv("DEFAULT_TEXT_MODEL"):
            self.default_text_model = os.getenv("DEFAULT_TEXT_MODEL")
        else:
            self.default_text_model = self.get("models.text_generation.small.0", "gpt2")
        
        if os.getenv("DEFAULT_CLASSIFICATION_MODEL"):
            self.default_classification_model = os.getenv("DEFAULT_CLASSIFICATION_MODEL")
        else:
            self.default_classification_model = self.get(
                "models.classification.sentiment.0", 
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        
        if os.getenv("DEFAULT_QA_MODEL"):
            self.default_qa_model = os.getenv("DEFAULT_QA_MODEL")
        else:
            self.default_qa_model = self.get(
                "models.question_answering.general.0",
                "distilbert-base-cased-distilled-squad"
            )
        
        # Generation parameters
        self.max_length = int(os.getenv("DEFAULT_MAX_LENGTH", self.get("default_settings.generation.max_length", 100)))
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", self.get("default_settings.generation.temperature", 0.7)))
        self.top_p = float(os.getenv("DEFAULT_TOP_P", self.get("default_settings.generation.top_p", 0.9)))
        
        # Device settings
        self.device = os.getenv("DEVICE", "auto")
        
        # Cache settings
        self.cache_dir = os.getenv("HF_HOME", self.get("cache_settings.model_cache_dir", "models/.cache"))
        
        # Quantization settings
        self.use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
        self.quantization_bits = int(os.getenv("QUANTIZATION_BITS", self.get("quantization.default_bits", 4)))
        
        # API keys
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Gradio settings
        self.gradio_port = int(os.getenv("GRADIO_PORT", 7860))
        self.gradio_share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if k.isdigit():
                    value = value[int(k)]
                else:
                    value = value[k]
            return value
        except (KeyError, IndexError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to file."""
        if file_path is None:
            file_path = self.config_file
        
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_model_list(self, task_type: str, category: str = None) -> list:
        """Get list of models for a specific task."""
        if category:
            return self.get(f"models.{task_type}.{category}", [])
        else:
            task_models = self.get(f"models.{task_type}", {})
            all_models = []
            for models in task_models.values():
                if isinstance(models, list):
                    all_models.extend(models)
            return all_models
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.get("default_settings.generation.do_sample", True),
            "num_return_sequences": self.get("default_settings.generation.num_return_sequences", 1)
        }
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration."""
        return {
            "use_quantization": self.use_quantization,
            "bits": self.quantization_bits,
            "compute_dtype": self.get("quantization.compute_dtype", "float16"),
            "quant_type": self.get("quantization.quant_type", "nf4")
        }
    
    def get_hardware_requirements(self, level: str = "recommended") -> Dict[str, Any]:
        """Get hardware requirements for a specific level."""
        return self.get(f"hardware_requirements.{level}", {})
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("🔧 Hugging Face Workspace Configuration")
        print("=" * 50)
        print(f"Default Text Model: {self.default_text_model}")
        print(f"Default Classification Model: {self.default_classification_model}")
        print(f"Default QA Model: {self.default_qa_model}")
        print(f"Device: {self.device}")
        print(f"Cache Directory: {self.cache_dir}")
        print(f"Use Quantization: {self.use_quantization}")
        if self.use_quantization:
            print(f"Quantization Bits: {self.quantization_bits}")
        print(f"Generation Max Length: {self.max_length}")
        print(f"Generation Temperature: {self.temperature}")
        print(f"Gradio Port: {self.gradio_port}")
        print(f"Log Level: {self.log_level}")
        
        if hasattr(self, 'hf_token') and self.hf_token:
            print("✅ Hugging Face Token: Configured")
        else:
            print("❌ Hugging Face Token: Not configured")
        
        if self.wandb_api_key:
            print("✅ Weights & Biases: Configured")
        else:
            print("❌ Weights & Biases: Not configured")


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Example usage
    config.print_summary()
    
    print("\n📋 Available Models:")
    print("Text Generation:", config.get_model_list("text_generation"))
    print("Classification:", config.get_model_list("classification"))
    print("Question Answering:", config.get_model_list("question_answering"))
    
    print("\n⚙️ Generation Parameters:")
    print(config.get_generation_params())
    
    print("\n💻 Hardware Requirements (Recommended):")
    print(config.get_hardware_requirements("recommended"))