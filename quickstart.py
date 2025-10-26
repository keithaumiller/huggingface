#!/usr/bin/env python3
"""
Quick start script for Hugging Face workspace.
Provides an interactive menu to get started quickly.
"""

import os
import sys
import subprocess
from utils.config import config

def print_header():
    print("🤗 Hugging Face LLM Workspace")
    print("=" * 40)
    print()

def print_menu():
    print("What would you like to do?")
    print()
    print("1. 📝 Generate Text (Quick)")
    print("2. 🎨 Launch Web Interface")
    print("3. 🔬 Run Text Generation Examples")
    print("4. 📊 Model Comparison")
    print("5. ⚙️  View Configuration")
    print("6. 🧪 Test Model Loading")
    print("7. 📚 View Available Models")
    print("8. 🚀 Setup Environment")
    print("9. ❓ Help & Documentation")
    print("0. 🚪 Exit")
    print()

def quick_generate():
    print("📝 Quick Text Generation")
    print("-" * 25)
    
    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        prompt = "The future of artificial intelligence is"
    
    model = input(f"Model (default: {config.default_text_model}): ").strip()
    if not model:
        model = config.default_text_model
    
    max_length = input(f"Max length (default: {config.max_length}): ").strip()
    if not max_length:
        max_length = str(config.max_length)
    
    print("\n🎯 Generating text...")
    
    cmd = [
        sys.executable, "main.py",
        "--prompt", prompt,
        "--model", model,
        "--max-length", max_length
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error running text generation")
    except KeyboardInterrupt:
        print("\n⏹️ Generation cancelled")

def launch_interface():
    print("🎨 Launching Web Interface...")
    print("This will open a web browser with interactive AI tools.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        cmd = [sys.executable, "examples/gradio_interface.py"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error launching interface")
    except KeyboardInterrupt:
        print("\n⏹️ Interface stopped")

def run_examples():
    print("🔬 Text Generation Examples")
    print("-" * 30)
    print("1. Basic text generation examples")
    print("2. Model loading and comparison")
    print("3. Advanced generation techniques")
    print()
    
    choice = input("Choose example (1-3): ").strip()
    
    scripts = {
        "1": "examples/text_generation.py",
        "2": "examples/model_loading.py",
        "3": "utils/model_utils.py"
    }
    
    script = scripts.get(choice)
    if script:
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error running {script}")
        except KeyboardInterrupt:
            print("\n⏹️ Example stopped")
    else:
        print("❌ Invalid choice")

def model_comparison():
    print("📊 Model Comparison")
    print("-" * 20)
    
    prompt = input("Enter prompt for comparison: ").strip()
    if not prompt:
        prompt = "The future of technology is"
    
    models = input("Models (comma-separated, default: gpt2,distilgpt2): ").strip()
    if not models:
        models = "gpt2,distilgpt2"
    
    print("\n🔍 Comparing models...")
    
    # Use the comparison function from gradio_interface
    try:
        from examples.gradio_interface import compare_models
        result = compare_models(prompt, models)
        print("\n" + "=" * 60)
        print(result)
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

def view_config():
    print("⚙️ Configuration Summary")
    print("-" * 25)
    config.print_summary()

def test_models():
    print("🧪 Testing Model Loading")
    print("-" * 25)
    
    models_to_test = ["gpt2", "distilgpt2"]
    
    for model_name in models_to_test:
        try:
            print(f"\n📋 Testing {model_name}...")
            from utils.model_utils import print_model_summary
            print_model_summary(model_name)
            print("✅ Test passed")
        except Exception as e:
            print(f"❌ Test failed: {e}")

def show_models():
    print("📚 Available Models")
    print("-" * 20)
    
    print("\n🔤 Text Generation Models:")
    text_models = config.get_model_list("text_generation")
    for i, model in enumerate(text_models, 1):
        print(f"  {i}. {model}")
    
    print("\n📊 Classification Models:")
    class_models = config.get_model_list("classification")
    for i, model in enumerate(class_models, 1):
        print(f"  {i}. {model}")
    
    print("\n❓ Question Answering Models:")
    qa_models = config.get_model_list("question_answering")
    for i, model in enumerate(qa_models, 1):
        print(f"  {i}. {model}")

def setup_environment():
    print("🚀 Environment Setup")
    print("-" * 20)
    
    print("Running setup script...")
    try:
        subprocess.run(["./setup.sh"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Setup script failed")
        print("Try running: chmod +x setup.sh && ./setup.sh")
    except FileNotFoundError:
        print("❌ Setup script not found")

def show_help():
    print("📖 Help & Documentation")
    print("-" * 25)
    
    print("📁 Project Structure:")
    print("  • main.py - Basic text generation script")
    print("  • examples/ - Example scripts and interfaces")
    print("  • utils/ - Utility functions and configuration")
    print("  • config/ - Configuration files")
    print("  • models/ - Model cache directory")
    print()
    
    print("🔗 Useful Links:")
    print("  • Hugging Face Hub: https://huggingface.co")
    print("  • Transformers Docs: https://huggingface.co/docs/transformers")
    print("  • Gradio Docs: https://gradio.app/docs")
    print()
    
    print("💡 Quick Tips:")
    print("  • Edit .env file to add your API keys")
    print("  • Use smaller models (gpt2, distilgpt2) for testing")
    print("  • Enable quantization for large models to save memory")
    print("  • Check GPU availability with nvidia-smi")
    print()
    
    print("🆘 Troubleshooting:")
    print("  • Out of memory? Try quantization or smaller models")
    print("  • Slow downloads? Set HF_HOME in .env")
    print("  • Import errors? Run: pip install -r requirements.txt")

def main():
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Enter your choice (0-9): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        print()  # Add space after input
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            quick_generate()
        elif choice == "2":
            launch_interface()
        elif choice == "3":
            run_examples()
        elif choice == "4":
            model_comparison()
        elif choice == "5":
            view_config()
        elif choice == "6":
            test_models()
        elif choice == "7":
            show_models()
        elif choice == "8":
            setup_environment()
        elif choice == "9":
            show_help()
        else:
            print("❌ Invalid choice. Please select 0-9.")
        
        print("\n" + "="*50 + "\n")
        input("Press Enter to continue...")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your installation and try again.")