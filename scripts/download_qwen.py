#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os
from pathlib import Path

def download_qwen():
    print("🚀 Starting Qwen2.5-3B-Instruct download...")
    print("=" * 60)
    
    repo_id = "Qwen/Qwen2.5-3B-Instruct"
    local_dir = "./qwen2.5-3b-instruct"
    
    print(f"📥 Downloading: {repo_id}")
    print(f"💾 Local directory: {local_dir}")
    
    try:
        print("\n📥 Downloading model files...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Qwen2.5-3B-Instruct Download Script")
    success = download_qwen()
    
    if success:
        print("\n🎉 Qwen2.5-3B-Instruct setup complete!")
    else:
        print("\n❌ Setup failed.")
