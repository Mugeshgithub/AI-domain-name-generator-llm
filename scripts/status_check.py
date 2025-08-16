#!/usr/bin/env python3
"""
Quick Status Check: Project Status
Shows current state without loading the model
"""

import os
import json
from pathlib import Path

def check_status():
    print("🔍 QUICK STATUS CHECK: AI Homework Project")
    print("=" * 50)
    
    # Check model files
    print("📁 MODEL STATUS:")
    base_model_path = "./qwen2.5-3b-instruct"
    if os.path.exists(base_model_path):
        print("   ✅ Base model exists (Qwen2.5-3B-Instruct)")
        
        # Check model size
        model_files = list(Path(base_model_path).rglob("*.safetensors"))
        total_size = sum(f.stat().st_size for f in model_files)
        size_gb = total_size / (1024**3)
        print(f"   📊 Model size: {size_gb:.1f} GB")
    else:
        print("   ❌ Base model missing")
    
    # Check fine-tuned model
    ft_model_path = "./fine_tuned_model"
    if os.path.exists(ft_model_path):
        print("   ✅ Fine-tuned model exists")
        ft_files = list(Path(ft_model_path).rglob("*"))
        print(f"   📊 Fine-tuned files: {len(ft_files)}")
    else:
        print("   ❌ Fine-tuned model does NOT exist")
    
    # Check training data
    print("\n📊 TRAINING DATA:")
    if os.path.exists("training_dataset.json"):
        with open("training_dataset.json", "r") as f:
            data = json.load(f)
        print(f"   ✅ Training dataset: {len(data)} examples")
        
        if len(data) < 10:
            print("   ⚠️  Very small dataset - may need more examples")
        elif len(data) < 100:
            print("   ⚠️  Small dataset - consider expanding")
        else:
            print("   ✅ Dataset size looks good")
    else:
        print("   ❌ Training dataset missing")
    
    # Check evaluation results
    print("\n📈 EVALUATION STATUS:")
    if os.path.exists("evaluation_report.json"):
        print("   ✅ Evaluation report exists")
    else:
        print("   ❌ No evaluation report found")
    
    if os.path.exists("model_comparison_report.json"):
        print("   ✅ Model comparison report exists")
    else:
        print("   ❌ No model comparison report found")
    
    # Check scripts
    print("\n🔧 SCRIPTS STATUS:")
    scripts_dir = "./scripts"
    if os.path.exists(scripts_dir):
        scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
        print(f"   ✅ Scripts directory: {len(scripts)} Python files")
        
        # Check key scripts
        key_scripts = ["fine_tune_lora.py", "evaluate_improvements.py", "quick_evaluation.py"]
        for script in key_scripts:
            if os.path.exists(os.path.join(scripts_dir, script)):
                print(f"   ✅ {script}")
            else:
                print(f"   ❌ {script} missing")
    else:
        print("   ❌ Scripts directory missing")
    
    # Overall status
    print("\n🎯 OVERALL STATUS:")
    
    if os.path.exists(ft_model_path):
        print("   🎉 FINE-TUNING COMPLETE!")
        print("   ✅ Ready for evaluation and comparison")
    else:
        print("   ⚠️  FINE-TUNING NOT COMPLETE")
        print("   ❌ Missing fine-tuned model")
        print("   ❌ No model comparison possible")
        print("   ❌ Core requirement not met")
    
    print(f"\n⏰ TIME ESTIMATES:")
    print(f"   Model Loading: 10-15 seconds (one-time)")
    print(f"   Generation: 10-30 seconds per prompt")
    print(f"   Full Evaluation: 5-10 minutes")
    print(f"   Fine-tuning: 30-60 minutes (not done)")

if __name__ == "__main__":
    check_status()
