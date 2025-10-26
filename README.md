# 🤗 Hugging Face LLM Workspace

A comprehensive development environment for working with Hugging Face large language models, transformers, and AI applications.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Copy environment template
   cp .env.template .env
   # Edit .env with your API keys and preferences
   ```

2. **Run Basic Example**
   ```bash
   python main.py --prompt "The future of AI is" --model gpt2
   ```

3. **Launch Interactive Interface**
   ```bash
   python examples/gradio_interface.py
   ```

## 📁 Project Structure

```
huggingface/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Main entry point
├── .env.template            # Environment variables template
├── config/
│   └── config.json          # Configuration settings
├── examples/
│   ├── text_generation.py   # Text generation examples
│   ├── model_loading.py     # Model loading and comparison
│   └── gradio_interface.py  # Web interface with Gradio
├── utils/
│   ├── config.py            # Configuration management
│   └── model_utils.py       # Utility functions
├── models/                  # Model cache directory
└── results/                 # Output and results directory
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for larger models)
- Optional: NVIDIA GPU with CUDA support

### Step-by-Step Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Copy the environment template
   cp .env.template .env
   
   # Edit .env file with your settings
   nano .env
   ```

3. **Essential Environment Variables**
   ```bash
   # Get your token from https://huggingface.co/settings/tokens
   HUGGINGFACE_HUB_TOKEN=your_token_here
   
   # Optional: For experiment tracking
   WANDB_API_KEY=your_wandb_key_here
   
   # Device configuration
   DEVICE=auto  # or cpu, cuda, cuda:0, etc.
   ```

4. **Test Installation**
   ```bash
   python -c "from transformers import pipeline; print('✅ Installation successful!')"
   ```

## 🎯 Usage Examples

### Basic Text Generation

```python
from transformers import pipeline

# Create a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator("The future of artificial intelligence", 
                  max_length=100, temperature=0.7)
print(result[0]['generated_text'])
```

### Using the Workspace Scripts

1. **Simple Text Generation**
   ```bash
   python main.py --prompt "Once upon a time" --max-length 150
   ```

2. **Model Comparison**
   ```bash
   python examples/text_generation.py
   ```

3. **Interactive Web Interface**
   ```bash
   python examples/gradio_interface.py --port 7860
   ```

4. **Model Analysis**
   ```bash
   python utils/model_utils.py
   ```

## 🤖 Supported Models

### Text Generation Models
- **Small**: `gpt2`, `distilgpt2`
- **Medium**: `gpt2-medium`, `microsoft/DialoGPT-medium`
- **Large**: `gpt2-large`, `gpt2-xl`

### Classification Models
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base`

### Question Answering
- **General**: `distilbert-base-cased-distilled-squad`
- **Conversational**: `deepset/roberta-base-squad2`

### Named Entity Recognition
- **General**: `dbmdz/bert-large-cased-finetuned-conll03-english`

## 🎨 Web Interface Features

The Gradio interface (`examples/gradio_interface.py`) provides:

- **Text Generation**: Interactive text generation with parameter controls
- **Text Classification**: Sentiment and emotion analysis
- **Question Answering**: Ask questions about any text
- **Model Comparison**: Compare outputs from multiple models
- **Real-time Results**: Instant feedback and results

### Launching the Interface

```bash
# Launch all interfaces
python examples/gradio_interface.py

# Launch specific interface
python examples/gradio_interface.py --interface generation
python examples/gradio_interface.py --interface classification
python examples/gradio_interface.py --interface qa

# Share publicly (creates public link)
python examples/gradio_interface.py --share
```

## ⚙️ Configuration

### Configuration Files

1. **`.env`** - Environment variables and API keys
2. **`config/config.json`** - Model lists, default parameters, hardware requirements

### Key Configuration Options

```python
from utils.config import config

# Print current configuration
config.print_summary()

# Get model lists
text_models = config.get_model_list("text_generation")
classification_models = config.get_model_list("classification")

# Get default parameters
gen_params = config.get_generation_params()
```

## 🔧 Advanced Features

### Quantization (for Large Models)

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    quantization_config=quantization_config
)
```

### Model Benchmarking

```python
from utils.model_utils import benchmark_model, print_model_summary

# Analyze a model
print_model_summary("gpt2")

# Benchmark model performance
results = benchmark_model("gpt2", num_runs=3)
```

### Custom Model Loading

```python
from examples.model_loading import ModelLoader

loader = ModelLoader()
model, tokenizer = loader.load_causal_lm("gpt2", quantize=True)

# Get model information
info = loader.get_model_info("gpt2")
print(f"Model size: {info['model_size_mb']:.1f} MB")
```

## 📊 Experiment Tracking

### Weights & Biases Integration

```python
import wandb
from utils.config import config

# Initialize W&B (configure WANDB_API_KEY in .env)
wandb.init(project="huggingface-experiments")

# Log model outputs
wandb.log({"generated_text": generated_text, "prompt": prompt})
```

### Saving Results

```python
from utils.model_utils import save_results, load_results

# Save experiment results
results = {"model": "gpt2", "prompt": prompt, "output": output}
save_results(results, "experiment_1.json")

# Load previous results
previous_results = load_results("experiment_1.json")
```

## 🚀 Performance Tips

### For Large Models
1. **Use Quantization**: Enable 4-bit or 8-bit quantization for large models
2. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi`
3. **Batch Processing**: Process multiple inputs together when possible

### For Better Results
1. **Prompt Engineering**: Craft clear, specific prompts
2. **Temperature Control**: Lower temperature (0.1-0.3) for focused output, higher (0.7-1.0) for creativity
3. **Model Selection**: Choose appropriate model size for your task

### Environment Variables for Performance

```bash
# Enable optimized memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use optimized CUDA kernels
export CUDA_LAUNCH_BLOCKING=0

# Cache models locally
export HF_HOME=/workspaces/huggingface/models/.cache
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use quantization or smaller batch sizes
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       device_map="auto"
   )
   ```

2. **Model Download Issues**
   ```bash
   # Set Hugging Face token for private models
   huggingface-cli login
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **Gradio Interface Not Accessible**
   ```bash
   # Check if port is available
   python examples/gradio_interface.py --port 7861
   ```

### Getting Help

- Check the [Hugging Face Documentation](https://huggingface.co/docs)
- Visit [Transformers GitHub](https://github.com/huggingface/transformers)
- Join the [Hugging Face Discord](https://discord.gg/hugging-face)

## 📚 Learning Resources

### Tutorials and Guides
- [Hugging Face Course](https://huggingface.co/course)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://gradio.app/docs)

### Example Notebooks
- Text generation with different models
- Fine-tuning for custom tasks
- Model comparison and evaluation
- Prompt engineering techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co) for their incredible transformers library
- [Gradio](https://gradio.app) for the easy-to-use interface framework
- The open-source AI community for making these tools accessible

---

**Happy experimenting with Hugging Face models! 🤗🚀**