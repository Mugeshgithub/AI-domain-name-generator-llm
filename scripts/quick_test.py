#!/usr/bin/env python3
"""
Quick Test: Base Model Performance
Fast evaluation to check if fine-tuning is needed
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_evaluation():
    print("🚀 QUICK EVALUATION: Base Model Performance")
    print("=" * 50)
    
    # Load model (this is the slow part)
    print("📥 Loading model...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained('./qwen2.5-3b-instruct')
    model = AutoModelForCausalLM.from_pretrained(
        './qwen2.5-3b-instruct',
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f} seconds")
    
    # Check device
    device = next(model.parameters()).device
    print(f"🔧 Model device: {device}")
    
    # Quick test cases
    test_cases = [
        "Generate 3 domain names for: coffee shop",
        "Generate 3 domain names for: tech startup"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {prompt}")
        print("   ⏳ Generating... (this may take 10-30 seconds)")
        
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
        
        # Fix device issue - move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50  # Limit generation length
            )
            
        generation_time = time.time() - start_time
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract domains
        domains = []
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('.com' in line or '.net' in line or '.org' in line):
                domains.append(line)
        
        quality_score = min(10, len(domains) * 2)
        
        results.append({
            "prompt": prompt,
            "domains": domains,
            "generation_time": generation_time,
            "quality_score": quality_score
        })
        
        print(f"   ✅ Done in {generation_time:.1f}s")
        print(f"   📊 Quality: {quality_score}/10")
        print(f"   🏷️  Domains: {domains[:3]}")
    
    # Summary
    total_time = sum(r['generation_time'] for r in results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Generation Time: {total_time:.1f}s")
    print(f"   Average Quality: {avg_quality:.1f}/10")
    print(f"   Model Load Time: {load_time:.1f}s")
    
    print(f"\n🎯 STATUS:")
    if avg_quality >= 7:
        print("   ✅ Base model performs well - fine-tuning may not be necessary")
    else:
        print("   ⚠️  Base model needs improvement - fine-tuning recommended")
    
    print(f"\n🔍 FINE-TUNING STATUS:")
    print("   ❌ Fine-tuned model does NOT exist")
    print("   ❌ No training has been executed")
    print("   ❌ No model comparison possible")
    print("   ✅ Base model evaluation complete")
    
    print(f"\n⏰ TIME BREAKDOWN:")
    print(f"   Model Loading: {load_time:.1f}s (one-time cost)")
    print(f"   Generation: {total_time:.1f}s (per-run cost)")
    print(f"   Total: {load_time + total_time:.1f}s")

if __name__ == "__main__":
    quick_evaluation()
