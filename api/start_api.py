#!/usr/bin/env python3
"""
Start the Domain Name Generation API
Simple launcher script with status information
"""

import os
import sys
import subprocess
import time

def start_api():
    """Start the API server"""
    
    print("🚀 STARTING DOMAIN NAME GENERATION API")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found!")
        print("   Please run this script from the api/ directory")
        return
    
    # Check if requirements are installed
    print("📦 Checking requirements...")
    try:
        import fastapi
        import uvicorn
        import torch
        import transformers
        import peft
        print("   ✅ All required packages are installed")
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        print("   Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check if model files exist
    print("\n🔍 Checking model files...")
    base_model_path = "../qwen2.5-3b-instruct"
    fine_tuned_path = "../fine_tuned_model_stable"
    
    if os.path.exists(base_model_path):
        print(f"   ✅ Base model found: {base_model_path}")
    else:
        print(f"   ❌ Base model missing: {base_model_path}")
        print("   Please download the model first")
        return
    
    if os.path.exists(fine_tuned_path):
        print(f"   ✅ Fine-tuned model found: {fine_tuned_path}")
    else:
        print(f"   ⚠️  Fine-tuned model not found: {fine_tuned_path}")
        print("   Will use base model only")
    
    # Start the API
    print(f"\n🎯 Starting API server...")
    print(f"   🌐 URL: http://localhost:8000")
    print(f"   📚 Docs: http://localhost:8000/docs")
    print(f"   🔍 Health: http://localhost:8000/health")
    print(f"\n   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the API server
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print(f"\n🛑 API server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API: {e}")

if __name__ == "__main__":
    start_api()
