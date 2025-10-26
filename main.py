#!/usr/bin/env python3
"""
Main entry point for Hugging Face LLM experiments.
"""

import os
import argparse
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def main():
    """Main function to demonstrate basic Hugging Face LLM usage."""
    parser = argparse.ArgumentParser(description="Hugging Face LLM Experiments")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium", 
                       help="Model name from Hugging Face Hub")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Text prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        # Create a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Generate text
        print(f"\nPrompt: {args.prompt}")
        print("Generating response...")
        
        outputs = generator(
            args.prompt,
            max_length=args.max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print(f"\nGenerated text:")
        print("-" * 50)
        print(outputs[0]['generated_text'])
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model name is correct and you have internet connection.")


if __name__ == "__main__":
    main()