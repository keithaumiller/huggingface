#!/usr/bin/env python3
"""
Text generation examples using Hugging Face transformers.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import os
from typing import List, Optional


def load_model_and_tokenizer(model_name: str, quantize: bool = False):
    """Load a model and tokenizer with optional quantization."""
    print(f"Loading model: {model_name}")
    
    # Configure quantization if requested
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    return model, tokenizer


def generate_text(
    model, 
    tokenizer, 
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
    """Generate text using the loaded model."""
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(inputs)
        )
    
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def chat_with_model(model_name: str = "microsoft/DialoGPT-medium"):
    """Interactive chat with a conversational model."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    print(f"Chat started with {model_name}")
    print("Type 'quit' to exit")
    
    chat_history_ids = None
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        # Encode user input and append to chat history
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors='pt'
        )
        
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
        
        # Decode and print bot response
        bot_response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        print(f"Bot: {bot_response}")


def compare_models(prompt: str, models: List[str]):
    """Compare text generation from multiple models."""
    print(f"Comparing models on prompt: '{prompt}'")
    print("=" * 80)
    
    for model_name in models:
        try:
            print(f"\n--- {model_name} ---")
            model, tokenizer = load_model_and_tokenizer(model_name)
            
            generated_texts = generate_text(
                model, tokenizer, prompt,
                max_length=150,
                temperature=0.7
            )
            
            print(f"Generated: {generated_texts[0]}")
            
            # Clean up memory
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    # Example usage
    print("Hugging Face Text Generation Examples")
    print("====================================")
    
    # Simple text generation
    prompt = "The future of artificial intelligence is"
    model_name = "gpt2"  # Small model for testing
    
    print(f"1. Simple text generation with {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    generated = generate_text(model, tokenizer, prompt, max_length=100)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated[0]}")
    print()
    
    # Using pipeline (easier API)
    print("2. Using pipeline API")
    generator = pipeline(
        "text-generation", 
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    result = generator(
        prompt, 
        max_length=100, 
        num_return_sequences=1,
        temperature=0.7
    )
    print(f"Pipeline result: {result[0]['generated_text']}")
    
    # Uncomment to try interactive chat
    # print("\n3. Interactive chat")
    # chat_with_model()