# 🤗 Hugging Face LLM Workspace

A comprehensive, production-ready development environment for working with Hugging Face large language models, transformers, and AI applications. Features safe model downloading with interrupt handling, interactive web interfaces, and comprehensive tooling.

## 🚀 Quick Start

### **🎯 Interactive Menu (Recommended)**
```bash
python quickstart.py
```
*Access all features through an easy-to-use menu system*

### **📥 Model Explorer (Safe Downloads)**
```bash
python model_explorer.py
```
*Browse, download, and manage models with Ctrl+C safe interrupts*

### **⚡ Quick Commands**
```bash
# Basic text generation
python main.py --prompt "The future of AI is" --model gpt2

# Web interface
python examples/gradio_interface.py

# Setup everything
./setup.sh
```

## 📁 Project Structure

```
huggingface/
├── 📄 README.md                    # This comprehensive guide
├── 📋 requirements.txt             # Python dependencies
├── 🚀 quickstart.py               # Interactive menu system
├── 📥 model_explorer.py           # Safe model browser & downloader
├── 🧪 test_download_safety.py     # Download safety testing
├── 🐍 main.py                     # CLI text generation
├── ⚙️ setup.sh                    # Automated setup script
├── 🔒 .env.template               # Environment variables template
├── 📊 MODEL_GUIDE.md              # Model recommendations & info
├── 🛡️ DOWNLOAD_SAFETY.md          # Interrupt handling documentation
├── config/
│   └── 📊 config.json             # Model configurations
├── examples/
│   ├── 🎯 text_generation.py      # Text generation examples
│   ├── 🔄 model_loading.py        # Model loading & comparison
│   └── 🎨 gradio_interface.py     # Interactive web interfaces
├── utils/
│   ├── ⚙️ config.py               # Configuration management
│   └── 🔧 model_utils.py          # Utility functions & analysis
├── models/                        # Model cache directory
└── results/                       # Output and results directory
```

## 🛠️ Installation & Setup

### **⚡ One-Command Setup**
```bash
./setup.sh
```
*Automated setup with dependency checking and safety verification*

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

## ✨ **Key Features**

### 🛡️ **Safe Model Downloads**
- **Interrupt-safe downloading** - Press Ctrl+C anytime without corruption
- **Automatic resume capability** - Interrupted downloads continue seamlessly
- **Integrity verification** - Ensure downloaded models are complete
- **Smart caching** - No duplicate downloads, efficient storage

### 🎨 **Interactive Interfaces**
- **Web-based UI** with Gradio for all AI tasks
- **Interactive menu system** for easy navigation
- **Model comparison tools** side-by-side testing
- **Real-time generation** with parameter controls

### 🤖 **Comprehensive AI Tasks**
- **Text Generation** - GPT-style language models
- **Text Classification** - Sentiment, emotion, topic analysis
- **Question Answering** - Context-based Q&A systems
- **Named Entity Recognition** - Extract entities from text
- **Text Summarization** - Automatic content summarization

### 🔧 **Advanced Features**
- **Quantization support** for large models (4-bit, 8-bit)
- **Multi-GPU support** with automatic device mapping
- **Experiment tracking** with Weights & Biases integration
- **Configuration management** with environment variables
- **Model benchmarking** and performance analysis

## 🎯 Usage Examples

### **🚀 Interactive Menu System**
```bash
python quickstart.py
```
*One-stop access to all features with guided workflows*

### **📥 Safe Model Downloads**
```bash
python model_explorer.py
```
*Browse 100,000+ models, download safely with Ctrl+C interrupt support*

### **🎨 Web Interface**
```bash
python examples/gradio_interface.py
```
*Complete AI playground with text generation, classification, Q&A, and more*

### **⚡ Quick Commands**

#### Text Generation
```bash
# Basic generation
python main.py --prompt "The future of AI is" --model gpt2

# Advanced generation with parameters
python main.py --prompt "Once upon a time" --max-length 150 --temperature 0.8
```

#### Model Management
```bash
# Browse and download models safely
python model_explorer.py

# Check model integrity
python -c "
from model_explorer import ModelExplorer
explorer = ModelExplorer()
result = explorer.check_download_integrity('gpt2')
print('Complete:', result.get('complete', False))
"

# View available models
python -c "from utils.config import config; print(config.get_model_list('text_generation'))"
```

#### Advanced Features
```bash
# Model comparison
python examples/text_generation.py

# Comprehensive model analysis
python examples/model_loading.py

# Performance benchmarking
python utils/model_utils.py
```

3. **Interactive Web Interface**
   ```bash
   python examples/gradio_interface.py --port 7860
   ```

4. **Model Analysis**
   ```bash
   python utils/model_utils.py
   ```

## 🤖 Models & Downloads

### **📥 100,000+ Models Available**
Access the entire Hugging Face model hub through our safe download system:

```bash
python model_explorer.py
```

### **🏆 Most Popular Models (by downloads)**
- **openai-community/gpt2** - 10.6M downloads (550MB)
- **Qwen/Qwen2.5-7B-Instruct** - 7.6M downloads (~14GB)  
- **meta-llama/Llama-3.1-8B-Instruct** - 5.2M downloads (~16GB)
- **distilbert-base-uncased-finetuned-sst-2-english** - 5.2M downloads
- **facebook/bart-large-cnn** - 2.7M downloads

### **🟢 Recommended Starter Models (< 2GB total)**
```bash
# Download these first for testing
gpt2                    # 550MB - Basic text generation
distilgpt2             # 350MB - Faster text generation  
t5-small               # 242MB - Multi-task model
distilbert-base-uncased # 268MB - Classification tasks
```

### **🟡 Production Models (< 5GB total)**
```bash
# Upgrade to these for better performance  
gpt2-medium            # 1.5GB - Better text quality
t5-base                # 892MB - Better summarization
facebook/bart-large-cnn # 1.6GB - News summarization
```

### **🔍 Model Categories**
- **Text Generation**: GPT-2, GPT-Neo, OPT, Qwen, LLaMA
- **Classification**: BERT, RoBERTa, DistilBERT variants
- **Question Answering**: BERT, DeBERTa, RoBERTa fine-tuned
- **Summarization**: BART, T5, Pegasus
- **Translation**: mBart, Helsinki-NLP models
- **NER**: BERT, spaCy, domain-specific models

*See [MODEL_GUIDE.md](MODEL_GUIDE.md) for comprehensive model recommendations*

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

## 🛡️ Safety & Error Handling

Our workspace includes production-ready safety features:

- **⚡ Interrupt Safety**: Graceful handling of Ctrl+C during downloads
- **🔄 Resume Downloads**: Automatic resume for interrupted downloads  
- **✅ Integrity Checking**: Verify model completeness after download
- **🧹 Cleanup Tools**: Remove incomplete downloads safely
- **📊 Progress Tracking**: Real-time download progress with ETA

See [DOWNLOAD_SAFETY.md](DOWNLOAD_SAFETY.md) for technical details.

## 📈 Performance & Optimization

- **GPU Acceleration**: Automatic CUDA detection and usage
- **Memory Management**: Efficient model loading with garbage collection
- **Quantization Support**: 8-bit and 4-bit model compression
- **Batch Processing**: Optimize inference for multiple inputs
- **Caching**: Smart model and tokenizer caching

## 🧪 Testing & Development

```bash
# Test download safety features
python test_download_safety.py

# Validate model integrity
python -c "from model_explorer import ModelExplorer; ModelExplorer().check_all_models()"

# Monitor GPU usage
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

## 📚 Additional Resources

- **[Hugging Face Model Hub](https://huggingface.co/models)** - Browse 100,000+ models
- **[Transformers Documentation](https://huggingface.co/docs/transformers)** - Official API docs
- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - Our model recommendations
- **[DOWNLOAD_SAFETY.md](DOWNLOAD_SAFETY.md)** - Safety feature details

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

**🚀 Ready to explore AI? Start with `python quickstart.py` and dive into the world of large language models!**