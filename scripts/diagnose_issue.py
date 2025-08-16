#!/usr/bin/env python3
"""
Diagnostic Script: Why is model loading so slow?
"""

import os
import time
import torch
import psutil
from pathlib import Path

def diagnose_issue():
    print("🔍 DIAGNOSING SLOW MODEL LOADING")
    print("=" * 50)
    
    # Check system resources
    print("💻 SYSTEM RESOURCES:")
    memory = psutil.virtual_memory()
    print(f"   RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    print(f"   CPU: {psutil.cpu_count()} cores")
    
    # Check disk space
    disk = psutil.disk_usage('.')
    print(f"   Disk: {disk.free / (1024**3):.1f} GB free")
    
    # Check model files
    print("\n📁 MODEL FILES:")
    model_path = "./qwen2.5-3b-instruct"
    if os.path.exists(model_path):
        model_files = list(Path(model_path).rglob("*.safetensors"))
        print(f"   Found {len(model_files)} model files")
        
        for file in model_files:
            size_mb = file.stat().st_size / (1024**2)
            print(f"   {file.name}: {size_mb:.1f} MB")
    else:
        print("   ❌ Model directory not found")
        return
    
    # Check PyTorch configuration
    print("\n🔧 PYTORCH CONFIGURATION:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print("   ✅ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        print("   ✅ Using NVIDIA GPU (CUDA)")
    else:
        print("   ⚠️  Using CPU only (slow)")
    
    # Test small model loading
    print("\n🧪 TESTING SMALL MODEL LOADING:")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("   Loading tokenizer...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained('./qwen2.5-3b-instruct')
        tokenizer_time = time.time() - start
        print(f"   ✅ Tokenizer loaded in {tokenizer_time:.1f} seconds")
        
        print("   Loading model (this is the slow part)...")
        start = time.time()
        
        # Try with different device maps
        print("   Trying device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            './qwen2.5-3b-instruct',
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start
        print(f"   ✅ Model loaded in {load_time:.1f} seconds")
        
        # Check where model is loaded
        device = next(model.parameters()).device
        print(f"   Model device: {device}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    diagnose_issue()
