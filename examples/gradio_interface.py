#!/usr/bin/env python3
"""
Gradio interface examples for Hugging Face models.
"""

import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import json


class HuggingFaceInterface:
    """Class to create various Gradio interfaces for HF models."""
    
    def __init__(self):
        self.models = {}
        self.pipelines = {}
    
    def load_text_generator(self, model_name: str = "gpt2"):
        """Load a text generation model."""
        if model_name not in self.pipelines:
            print(f"Loading text generator: {model_name}")
            self.pipelines[model_name] = pipeline(
                "text-generation", 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        return self.pipelines[model_name]
    
    def load_classifier(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Load a classification model."""
        if model_name not in self.pipelines:
            print(f"Loading classifier: {model_name}")
            self.pipelines[model_name] = pipeline("sentiment-analysis", model=model_name)
        return self.pipelines[model_name]
    
    def load_qa_model(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """Load a question answering model."""
        if model_name not in self.pipelines:
            print(f"Loading QA model: {model_name}")
            self.pipelines[model_name] = pipeline("question-answering", model=model_name)
        return self.pipelines[model_name]


# Global interface instance
hf_interface = HuggingFaceInterface()


def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.7, model_name: str = "gpt2") -> str:
    """Generate text using the selected model."""
    try:
        generator = hf_interface.load_text_generator(model_name)
        
        result = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        return result[0]['generated_text']
    
    except Exception as e:
        return f"Error: {str(e)}"


def classify_text(text: str, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> str:
    """Classify text sentiment."""
    try:
        classifier = hf_interface.load_classifier(model_name)
        result = classifier(text)
        
        output = f"Label: {result[0]['label']}\n"
        output += f"Confidence: {result[0]['score']:.3f}"
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"


def answer_question(context: str, question: str) -> str:
    """Answer questions based on context."""
    try:
        qa_model = hf_interface.load_qa_model()
        result = qa_model(question=question, context=context)
        
        output = f"Answer: {result['answer']}\n"
        output += f"Confidence: {result['score']:.3f}\n"
        output += f"Start: {result['start']}, End: {result['end']}"
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"


def compare_models(prompt: str, models: List[str]) -> str:
    """Compare text generation from multiple models."""
    results = []
    models_list = [m.strip() for m in models.split(",") if m.strip()]
    
    for model_name in models_list:
        try:
            generated = generate_text(prompt, max_length=150, model_name=model_name)
            results.append(f"=== {model_name} ===\n{generated}\n")
        except Exception as e:
            results.append(f"=== {model_name} ===\nError: {str(e)}\n")
    
    return "\n".join(results)


def create_text_generation_interface():
    """Create a text generation interface."""
    with gr.Blocks(title="Text Generation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤗 Hugging Face Text Generation")
        gr.Markdown("Generate text using various pre-trained language models")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                model_dropdown = gr.Dropdown(
                    choices=["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"],
                    value="gpt2",
                    label="Model"
                )
                max_length_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Length"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    max_lines=20
                )
        
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_input, max_length_slider, temperature_slider, model_dropdown],
            outputs=output
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["The future of artificial intelligence is", "gpt2", 100, 0.7],
                ["Once upon a time in a distant galaxy", "gpt2", 150, 0.8],
                ["The benefits of renewable energy include", "distilgpt2", 120, 0.6]
            ],
            inputs=[prompt_input, model_dropdown, max_length_slider, temperature_slider]
        )
    
    return demo


def create_classification_interface():
    """Create a text classification interface."""
    with gr.Blocks(title="Text Classification", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Text Classification")
        gr.Markdown("Analyze sentiment and classify text")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Classify",
                    placeholder="Enter text to analyze...",
                    lines=4
                )
                model_dropdown = gr.Dropdown(
                    choices=[
                        "cardiffnlp/twitter-roberta-base-sentiment-latest",
                        "distilbert-base-uncased-finetuned-sst-2-english"
                    ],
                    value="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    label="Classification Model"
                )
                classify_btn = gr.Button("Classify", variant="primary")
            
            with gr.Column():
                classification_output = gr.Textbox(
                    label="Classification Result",
                    lines=5
                )
        
        classify_btn.click(
            fn=classify_text,
            inputs=[text_input, model_dropdown],
            outputs=classification_output
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["I love this new technology!"],
                ["This is absolutely terrible."],
                ["It's okay, nothing special."],
                ["Amazing breakthrough in AI research!"]
            ],
            inputs=[text_input]
        )
    
    return demo


def create_qa_interface():
    """Create a question answering interface."""
    with gr.Blocks(title="Question Answering", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ❓ Question Answering")
        gr.Markdown("Ask questions about any text context")
        
        with gr.Row():
            with gr.Column():
                context_input = gr.Textbox(
                    label="Context",
                    placeholder="Paste your text context here...",
                    lines=8
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What would you like to know?",
                    lines=2
                )
                qa_btn = gr.Button("Answer", variant="primary")
            
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=6
                )
        
        qa_btn.click(
            fn=answer_question,
            inputs=[context_input, question_input],
            outputs=answer_output
        )
        
        # Example context and questions
        example_context = """
        Hugging Face is an AI company that develops tools for building applications using machine learning.
        Founded in 2016, the company is based in New York and Paris. They are best known for their 
        Transformers library, which provides thousands of pre-trained models for natural language processing,
        computer vision, and audio tasks. The company offers both open-source tools and commercial services,
        making advanced AI accessible to developers and researchers worldwide.
        """
        
        gr.Examples(
            examples=[
                [example_context, "When was Hugging Face founded?"],
                [example_context, "Where is the company based?"],
                [example_context, "What is Hugging Face known for?"],
                [example_context, "What services does Hugging Face offer?"]
            ],
            inputs=[context_input, question_input]
        )
    
    return demo


def create_model_comparison_interface():
    """Create a model comparison interface."""
    with gr.Blocks(title="Model Comparison", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚖️ Model Comparison")
        gr.Markdown("Compare text generation from multiple models")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                models_input = gr.Textbox(
                    label="Models (comma-separated)",
                    value="gpt2, distilgpt2",
                    placeholder="gpt2, distilgpt2, microsoft/DialoGPT-medium"
                )
                compare_btn = gr.Button("Compare Models", variant="primary")
            
            with gr.Column():
                comparison_output = gr.Textbox(
                    label="Comparison Results",
                    lines=15,
                    max_lines=25
                )
        
        compare_btn.click(
            fn=compare_models,
            inputs=[prompt_input, models_input],
            outputs=comparison_output
        )
    
    return demo


def launch_interface(interface_type: str = "generation"):
    """Launch the selected interface."""
    if interface_type == "generation":
        demo = create_text_generation_interface()
    elif interface_type == "classification":
        demo = create_classification_interface()
    elif interface_type == "qa":
        demo = create_qa_interface()
    elif interface_type == "comparison":
        demo = create_model_comparison_interface()
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")
    
    return demo


def create_all_in_one_interface():
    """Create a comprehensive interface with all features."""
    with gr.Blocks(title="Hugging Face AI Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤗 Hugging Face AI Playground")
        gr.Markdown("Explore various AI models and tasks in one place")
        
        with gr.Tabs():
            with gr.TabItem("Text Generation"):
                gen_interface = create_text_generation_interface()
            
            with gr.TabItem("Text Classification"):
                class_interface = create_classification_interface()
            
            with gr.TabItem("Question Answering"):
                qa_interface = create_qa_interface()
            
            with gr.TabItem("Model Comparison"):
                comp_interface = create_model_comparison_interface()
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hugging Face Gradio Interfaces")
    parser.add_argument(
        "--interface", 
        choices=["generation", "classification", "qa", "comparison", "all"],
        default="all",
        help="Which interface to launch"
    )
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    
    args = parser.parse_args()
    
    if args.interface == "all":
        demo = create_all_in_one_interface()
    else:
        demo = launch_interface(args.interface)
    
    print(f"Launching {args.interface} interface...")
    demo.launch(
        share=args.share, 
        server_port=args.port,
        server_name="0.0.0.0"  # Allow external access in dev container
    )