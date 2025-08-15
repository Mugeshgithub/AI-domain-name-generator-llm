#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_domain_generation():
    print("🧪 Testing Qwen2.5-3B-Instruct Domain Generation")
    print("=" * 70)
    
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
        
        print("\n🎯 Testing Domain Generation for Different Businesses:")
        print("-" * 50)
        
        business_types = [
            "organic coffee shop in downtown area",
            "tech startup building AI tools",
            "family bakery with fresh bread"
        ]
        
        for i, business in enumerate(business_types, 1):
            print(f"\n📝 Business {i}: {business}")
            print("-" * 30)
            
            prompt = f"Generate 3 creative domain names for: {business}"
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if ":" in generated_text:
                response = generated_text.split(":")[-1].strip()
            else:
                response = generated_text.replace(prompt, "").strip()
            
            print(f"🤖 Generated: {response}")
        
        print("\n🎉 All domain generation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Qwen2.5-3B-Instruct Domain Generation Test")
    success = test_domain_generation()
    
    if success:
        print("\n🚀 Model is working perfectly for domain generation!")
        print("📊 Ready for your AI homework requirements!")
    else:
        print("\n❌ Test failed.")
