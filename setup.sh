#!/bin/bash

# Hugging Face Environment Setup Script
# This script helps set up the Hugging Face development environment

set -e  # Exit on error

echo "🤗 Setting up Hugging Face LLM Workspace..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_info "Python version: $PYTHON_VERSION"
        
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            print_status "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
            print_info "GPU: $line MB"
        done
    else
        print_warning "No NVIDIA GPU detected. Using CPU mode."
    fi
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Set up environment file
setup_environment() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_status "Created .env file from template"
            print_warning "Please edit .env file with your API keys and preferences"
        else
            print_error ".env.template not found"
        fi
    else
        print_info ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    directories=("models" "results" "data" "logs")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    python3 -c "
import sys
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    
    import transformers
    print('✅ Transformers:', transformers.__version__)
    
    import gradio as gr
    print('✅ Gradio:', gr.__version__)
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print('✅ CUDA available:', torch.cuda.get_device_name(0))
    else:
        print('⚠️  CUDA not available - using CPU mode')
    
    print('🎉 All core packages installed successfully!')
    
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Run a quick model test
quick_test() {
    print_info "Running quick model test..."
    
    python3 -c "
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    # Test with a small model
    generator = pipeline('text-generation', model='gpt2', max_length=50)
    result = generator('Hello world', max_length=20, num_return_sequences=1)
    print('✅ Model test successful!')
    print('Generated text:', result[0]['generated_text'])
except Exception as e:
    print('❌ Model test failed:', e)
    raise
"
    
    if [ $? -eq 0 ]; then
        print_status "Quick model test passed"
    else
        print_error "Quick model test failed"
        exit 1
    fi
}

# Display usage information
show_usage() {
    echo
    print_info "🚀 Setup complete! Here's how to get started:"
    echo
    echo "1. Basic text generation:"
    echo "   python main.py --prompt 'Hello world' --model gpt2"
    echo
    echo "2. Launch web interface:"
    echo "   python examples/gradio_interface.py"
    echo
    echo "3. Run example scripts:"
    echo "   python examples/text_generation.py"
    echo "   python examples/model_loading.py"
    echo
    echo "4. Configure your environment:"
    echo "   nano .env  # Add your API keys"
    echo
    echo "5. View configuration:"
    echo "   python utils/config.py"
    echo
    print_status "Happy experimenting with Hugging Face! 🤗"
}

# Main execution
main() {
    echo "Starting Hugging Face workspace setup..."
    echo
    
    check_python
    check_cuda
    install_dependencies
    setup_environment
    create_directories
    test_installation
    quick_test
    show_usage
    
    echo
    print_status "Setup completed successfully!"
}

# Run setup with command line options
case "${1:-}" in
    --quick)
        print_info "Running quick setup (dependencies only)..."
        check_python
        install_dependencies
        ;;
    --test)
        print_info "Running tests only..."
        test_installation
        quick_test
        ;;
    --help|-h)
        echo "Hugging Face Setup Script"
        echo
        echo "Usage: $0 [option]"
        echo "Options:"
        echo "  --quick    Quick setup (install dependencies only)"
        echo "  --test     Run tests only"
        echo "  --help     Show this help message"
        echo
        ;;
    *)
        main
        ;;
esac