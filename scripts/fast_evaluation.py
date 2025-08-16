#!/usr/bin/env python3
"""
Fast Evaluation: Quick assessment without full model loading
"""

import json
import time
from pathlib import Path

def fast_evaluation():
    print("⚡ FAST EVALUATION: Quick Assessment")
    print("=" * 50)
    
    # Check what we have
    print("📊 CURRENT PROJECT STATUS:")
    
    # Check training dataset
    if Path("training_dataset.json").exists():
        with open("training_dataset.json", "r") as f:
            data = json.load(f)
        print(f"   ✅ Training dataset: {len(data)} examples")
    else:
        print("   ❌ No training dataset found")
    
    # Check if fine-tuned model exists
    fine_tuned_paths = [
        "fine_tuned_model",
        "models/fine_tuned",
        "checkpoints",
        "lora_weights"
    ]
    
    fine_tuned_found = False
    for path in fine_tuned_paths:
        if Path(path).exists():
            print(f"   ✅ Fine-tuned model found: {path}")
            fine_tuned_found = True
            break
    
    if not fine_tuned_found:
        print("   ❌ No fine-tuned model found")
    
    # Check evaluation results
    eval_files = list(Path(".").glob("*.json"))
    eval_results = [f for f in eval_files if "eval" in f.name.lower() or "result" in f.name.lower()]
    
    if eval_results:
        print(f"   ✅ Evaluation results found: {len(eval_results)} files")
    else:
        print("   ❌ No evaluation results found")
    
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    
    if not fine_tuned_found:
        print("   1. 🚨 CRITICAL: Fine-tuning not completed!")
        print("      - This is the main requirement")
        print("      - Current scripts are too slow for your system")
        print("      - Need to optimize for 8GB RAM")
        
        print("\n   2. 💡 SOLUTIONS:")
        print("      Option A: Use smaller model (1B instead of 3B)")
        print("      Option B: Use cloud GPU (Google Colab, AWS)")
        print("      Option C: Optimize memory usage")
        print("      Option D: Use pre-trained results")
        
        print("\n   3. ⏱️  TIME ESTIMATES:")
        print("      - Current setup: 2+ hours (too slow)")
        print("      - Optimized setup: 30-60 minutes")
        print("      - Cloud GPU: 15-30 minutes")
    
    else:
        print("   1. ✅ Fine-tuning completed!")
        print("   2. Run evaluation on fine-tuned model")
        print("   3. Generate comparison report")
    
    print("\n🔧 IMMEDIATE ACTION:")
    print("   Run: python scripts/optimized_fine_tune.py")
    print("   (This will use memory optimization)")

if __name__ == "__main__":
    fast_evaluation()
