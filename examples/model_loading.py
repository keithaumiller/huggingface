#!/usr/bin/env python3
"""
Examples for loading and working with different types of Hugging Face models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    pipeline,
    BitsAndBytesConfig
)
import json
from typing import Dict, List, Any


class ModelLoader:
    """Helper class for loading different types of models."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.models = {}
        self.tokenizers = {}
    
    def load_causal_lm(self, model_name: str, quantize: bool = False):
        """Load a causal language model (GPT-style)."""
        print(f"Loading causal LM: {model_name}")
        
        quantization_config = None
        if quantize and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=self.device if torch.cuda.is_available() else None
        )
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        return model, tokenizer
    
    def load_classification_model(self, model_name: str):
        """Load a sequence classification model."""
        print(f"Loading classification model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        return model, tokenizer
    
    def load_qa_model(self, model_name: str):
        """Load a question answering model."""
        print(f"Loading QA model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        return model, tokenizer
    
    def load_token_classification_model(self, model_name: str):
        """Load a token classification model (NER, etc.)."""
        print(f"Loading token classification model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        return model, tokenizer
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return {"error": "Model not loaded"}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        info = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "tokenizer_type": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        }
        
        return info


def demonstrate_text_generation():
    """Demonstrate text generation with different models."""
    print("=== Text Generation Examples ===")
    
    loader = ModelLoader()
    
    # Small models for demonstration
    models_to_try = [
        "gpt2",
        "distilgpt2",
    ]
    
    prompt = "The benefits of artificial intelligence include"
    
    for model_name in models_to_try:
        try:
            print(f"\n--- {model_name} ---")
            model, tokenizer = loader.load_causal_lm(model_name)
            
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            
            # Show model info
            info = loader.get_model_info(model_name)
            print(f"Parameters: {info['num_parameters']:,}")
            print(f"Size: {info['model_size_mb']:.1f} MB")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def demonstrate_classification():
    """Demonstrate text classification."""
    print("\n=== Text Classification Examples ===")
    
    # Use pipeline for simplicity
    classifier = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    texts = [
        "I love using Hugging Face transformers!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
        "Machine learning is fascinating and powerful."
    ]
    
    for text in texts:
        result = classifier(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
        print()


def demonstrate_question_answering():
    """Demonstrate question answering."""
    print("\n=== Question Answering Examples ===")
    
    qa_pipeline = pipeline("question-answering")
    
    context = """
    Hugging Face is a company that develops tools for building applications using machine learning.
    They are particularly known for their transformers library, which provides pre-trained models
    for natural language processing tasks. The company was founded in 2016 and is based in New York.
    They offer both open-source tools and commercial services for AI development.
    """
    
    questions = [
        "When was Hugging Face founded?",
        "Where is Hugging Face based?",
        "What is Hugging Face known for?",
        "What type of services does Hugging Face offer?"
    ]
    
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.3f})")
        print()


def demonstrate_named_entity_recognition():
    """Demonstrate named entity recognition."""
    print("\n=== Named Entity Recognition Examples ===")
    
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    
    texts = [
        "Apple Inc. is planning to open a new store in San Francisco next year.",
        "Elon Musk founded SpaceX in 2002 and Tesla in 2003.",
        "The meeting between Joe Biden and Emmanuel Macron took place in Paris."
    ]
    
    for text in texts:
        entities = ner_pipeline(text)
        print(f"Text: {text}")
        print("Entities:")
        for entity in entities:
            print(f"  - {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.3f})")
        print()


def model_comparison_report(models: List[str], prompt: str):
    """Generate a comparison report for multiple models."""
    print(f"\n=== Model Comparison Report ===")
    print(f"Prompt: '{prompt}'")
    print("=" * 80)
    
    loader = ModelLoader()
    results = {}
    
    for model_name in models:
        try:
            print(f"\nTesting {model_name}...")
            model, tokenizer = loader.load_causal_lm(model_name)
            
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            info = loader.get_model_info(model_name)
            
            results[model_name] = {
                "generated_text": generated_text,
                "parameters": info['num_parameters'],
                "size_mb": info['model_size_mb']
            }
            
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    # Print comparison
    for model_name, result in results.items():
        print(f"\n--- {model_name} ---")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Parameters: {result['parameters']:,}")
            print(f"Size: {result['size_mb']:.1f} MB")
            print(f"Generated: {result['generated_text']}")


if __name__ == "__main__":
    print("Hugging Face Model Loading Examples")
    print("==================================")
    
    # Demonstrate different model types
    demonstrate_text_generation()
    demonstrate_classification()
    demonstrate_question_answering()
    demonstrate_named_entity_recognition()
    
    # Model comparison
    small_models = ["gpt2", "distilgpt2"]
    comparison_prompt = "The future of technology is"
    model_comparison_report(small_models, comparison_prompt)