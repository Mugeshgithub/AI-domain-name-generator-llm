#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_qwen_domain_generation():
    print("🧪 Testing Qwen2.5-3B-Instruct for Domain Generation")
    print("=" * 60)
    
    try:
        print("📥 Loading Qwen2.5-3B-Instruct...")
        model_path = "./qwen2.5-3b-instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully!")
        
        print("\n🎯 Testing domain generation...")
        prompt = "Generate a domain name for a coffee shop:"
        print(f"📝 Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Generated: {generated_text}")
        
        print("\n🎉 Qwen2.5-3B-Instruct is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Qwen2.5-3B-Instruct Test Script")
    success = test_qwen_domain_generation()
    
    if success:
        print("\n🚀 Ready for fine-tuning and domain generation!")
        print("📊 Model size: ~6.2GB")
        print("🎯 Perfect for your AI homework!")
    else:
        print("\n❌ Test failed.")
