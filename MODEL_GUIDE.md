# 🤗 Recommended Models for Download

This document provides a curated list of models you can download and store locally for different use cases.

## 📊 Model Size Categories

### 🟢 Small Models (< 1GB) - Great for Development & Testing
| Model | Size | Task | Description |
|-------|------|------|-------------|
| `gpt2` | ~550MB | Text Generation | Original GPT-2 base model |
| `distilgpt2` | ~350MB | Text Generation | Smaller, faster version of GPT-2 |
| `microsoft/DialoGPT-small` | ~117MB | Conversational | Small conversational model |
| `distilbert-base-uncased` | ~268MB | Classification/QA | Lightweight BERT variant |
| `t5-small` | ~242MB | Text-to-Text | Small T5 for various tasks |

### 🟡 Medium Models (1-5GB) - Production Ready
| Model | Size | Task | Description |
|-------|------|------|-------------|
| `gpt2-medium` | ~1.5GB | Text Generation | Medium GPT-2 with better quality |
| `microsoft/DialoGPT-medium` | ~863MB | Conversational | Medium conversational model |
| `facebook/opt-1.3b` | ~2.6GB | Text Generation | Meta's OPT model |
| `t5-base` | ~892MB | Text-to-Text | Base T5 for summarization, QA |
| `facebook/bart-large-cnn` | ~1.6GB | Summarization | BART fine-tuned for news summarization |

### 🔴 Large Models (5GB+) - High Performance
| Model | Size | Task | Description |
|-------|------|------|-------------|
| `gpt2-large` | ~3.2GB | Text Generation | Large GPT-2 for high-quality text |
| `gpt2-xl` | ~6.4GB | Text Generation | Largest GPT-2 variant |
| `facebook/opt-6.7b` | ~13GB | Text Generation | Large OPT model |
| `EleutherAI/gpt-neo-2.7B` | ~10GB | Text Generation | Open-source large model |

## 🎯 Models by Use Case

### 📝 Creative Writing & Content Generation
```bash
# Download these models for creative writing
python model_explorer.py
# Then select option 4 and download:
gpt2                    # Start here (550MB)
gpt2-medium            # Better quality (1.5GB)
microsoft/DialoGPT-medium  # For dialogue (863MB)
```

### 📊 Text Analysis & Classification
```bash
# Sentiment analysis
cardiffnlp/twitter-roberta-base-sentiment-latest
distilbert-base-uncased-finetuned-sst-2-english

# Emotion detection
j-hartmann/emotion-english-distilroberta-base

# Topic classification
facebook/bart-large-mnli
```

### ❓ Question Answering Systems
```bash
# General QA
distilbert-base-cased-distilled-squad
deepset/roberta-base-squad2

# Conversational QA
microsoft/deberta-base
```

### 🔤 Named Entity Recognition (NER)
```bash
# General NER
dbmdz/bert-large-cased-finetuned-conll03-english
dslim/bert-base-NER

# Biomedical NER
d4data/biomedical-ner-all
```

### 📄 Text Summarization
```bash
# News summarization
facebook/bart-large-cnn

# General summarization
t5-base
t5-large

# Abstractive summarization
pegasus-xsum
```

### 🌍 Translation
```bash
# English to German
Helsinki-NLP/opus-mt-en-de

# English to French
Helsinki-NLP/opus-mt-en-fr

# Multilingual translation
facebook/mbart-large-50-many-to-many-mmt
```

## 🚀 Quick Start Downloads

### For Beginners (< 2GB total)
```bash
python model_explorer.py
# Download these in order:
1. gpt2                                    # (550MB) - Text generation
2. distilbert-base-uncased                 # (268MB) - Classification
3. distilbert-base-cased-distilled-squad   # (261MB) - QA
4. cardiffnlp/twitter-roberta-base-sentiment-latest # (499MB) - Sentiment
```

### For Developers (< 5GB total)
```bash
# Add these to the beginner set:
5. gpt2-medium                             # (1.5GB) - Better text generation
6. microsoft/DialoGPT-medium               # (863MB) - Conversations
7. t5-base                                 # (892MB) - Summarization
```

### For Production (< 15GB total)
```bash
# Add these for production use:
8. gpt2-large                              # (3.2GB) - High-quality generation
9. facebook/bart-large-cnn                 # (1.6GB) - News summarization
10. facebook/opt-1.3b                      # (2.6GB) - Alternative generation model
```

## 💾 Storage Management

### Check Available Space
```bash
# Check current usage
python model_explorer.py
# Select option 7 - Check storage usage
```

### Download Strategy
1. **Start Small**: Begin with models under 1GB
2. **Test First**: Always test models before committing to large downloads
3. **Monitor Space**: Keep track of your storage usage
4. **Clean Regularly**: Remove unused models periodically

### Cache Location
Models are stored in: `/workspaces/huggingface/models/.cache`

You can change this by editing the `HF_HOME` variable in your `.env` file.

## 🔄 Download Commands

### Using the Interactive Explorer
```bash
python model_explorer.py
# Follow the menu options to browse and download
```

### Direct Download in Python
```python
from model_explorer import ModelExplorer

explorer = ModelExplorer()

# Download a specific model
success = explorer.download_model("gpt2")

# Get model information first
info = explorer.get_model_info("gpt2")
print(f"Model size: {info['model_size']['estimated_size_mb']} MB")

# List downloaded models
downloaded = explorer.list_downloaded_models()
```

### Using Transformers Directly
```python
from transformers import AutoTokenizer, AutoModel

# This automatically downloads and caches the model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
```

## 🎛️ Advanced Options

### Force Re-download
```python
# Force fresh download (ignores cache)
explorer.download_model("gpt2", force_download=True)
```

### Custom Cache Directory
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "gpt2",
    cache_dir="/custom/cache/path"
)
```

### Quantized Downloads (Save Space)
```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# 4-bit quantization to save 75% space
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2-large", 
    quantization_config=quantization_config
)
```

## 🔍 Finding New Models

### Browse Hugging Face Hub
- Visit: https://huggingface.co/models
- Filter by: Task, Library, Size, Language
- Sort by: Downloads, Trending, Recent

### Search Programmatically
```python
from model_explorer import ModelExplorer

explorer = ModelExplorer()
models = explorer.search_models(task="text-generation", limit=20)
```

### Popular Model Collections
- **OpenAI Community**: `openai-community/*`
- **Meta AI**: `facebook/*` 
- **Google**: `google/*`
- **Microsoft**: `microsoft/*`
- **Hugging Face**: `HuggingFaceH4/*`

## 🚨 Important Notes

1. **Licensing**: Check model licenses before commercial use
2. **Rate Limits**: Respect Hugging Face download limits
3. **Storage**: Large models require significant disk space
4. **Memory**: Loading large models requires adequate RAM/GPU memory
5. **Internet**: Initial downloads require stable internet connection

## 🆘 Troubleshooting

### Download Failures
- Check internet connection
- Verify model name spelling
- Check available disk space
- Try again with `force_download=True`

### Out of Space
- Use quantized models
- Remove unused cached models
- Use smaller model variants
- Move cache to larger drive

### Memory Issues
- Use smaller models for testing
- Enable quantization
- Use CPU-only mode for very large models
- Close other applications