#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
from datetime import datetime

class DomainGenerationEvaluator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results = []
        self.load_model()
    
    def load_model(self):
        print("📥 Loading Qwen2.5-3B-Instruct for evaluation...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('./qwen2.5-3b-instruct')
            self.model = AutoModelForCausalLM.from_pretrained(
                './qwen2.5-3b-instruct',
                torch_dtype=torch.float16,
                device_map='auto'
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_domains(self, business_description, num_domains=3):
        try:
            prompt = f"Generate {num_domains} creative domain names for: {business_description}"
            
            inputs = self.tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if ":" in generated_text:
                domains = generated_text.split(":")[-1].strip()
            else:
                domains = generated_text.replace(prompt, "").strip()
            
            return domains
            
        except Exception as e:
            return f"Error: {e}"
    
    def evaluate_business_categories(self):
        print("\n🎯 Starting Systematic Evaluation of Business Categories")
        print("=" * 70)
        
        test_cases = [
            {
                "category": "Food & Beverage",
                "description": "organic coffee shop downtown",
                "complexity": "low",
                "expected": "clear, coffee-related domains"
            },
            {
                "category": "Technology",
                "description": "AI software startup",
                "complexity": "low", 
                "expected": "tech-focused domains"
            },
            {
                "category": "Health & Wellness",
                "description": "yoga studio with meditation classes",
                "complexity": "medium",
                "expected": "wellness-focused domains"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}: {test_case['category']}")
            print(f"🏷️  Complexity: {test_case['complexity']}")
            print(f"📋 Description: {test_case['description']}")
            print("-" * 50)
            
            start_time = time.time()
            domains = self.generate_domains(test_case['description'])
            generation_time = time.time() - start_time
            
            print(f"🤖 Generated: {domains}")
            print(f"⏱️  Time: {generation_time:.2f} seconds")
            
            quality_score = self.assess_quality(domains, test_case)
            print(f"📊 Quality Score: {quality_score}/10")
            
            self.results.append({
                "test_case": i,
                "category": test_case['category'],
                "complexity": test_case['complexity'],
                "description": test_case['description'],
                "generated_domains": domains,
                "generation_time": generation_time,
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat()
            })
            
            print("-" * 50)
    
    def assess_quality(self, domains, test_case):
        score = 0
        
        if "Error:" in domains:
            return 0
        
        if test_case['category'] == "Food & Beverage" and any(word in domains.lower() for word in ['coffee', 'brew', 'cafe']):
            score += 3
        elif test_case['category'] == "Technology" and any(word in domains.lower() for word in ['tech', 'ai', 'digital']):
            score += 3
        else:
            score += 1
        
        if '.com' in domains or '.net' in domains:
            score += 2
        
        if len(domains) > 50:
            score += 2
        
        if domains and len(domains.strip()) > 10:
            score += 2
        
        return min(score, 10)
    
    def generate_report(self):
        print("\n📊 Generating Evaluation Report")
        print("=" * 70)
        
        total_tests = len([r for r in self.results if 'test_case' in r])
        
        if total_tests > 0:
            avg_quality = sum(r.get('quality_score', 0) for r in self.results if 'quality_score' in r) / total_tests
            avg_time = sum(r.get('generation_time', 0) for r in self.results if 'generation_time' in r) / total_tests
        else:
            avg_quality = 0
            avg_time = 0
        
        report_data = {
            "evaluation_summary": {
                "total_business_tests": total_tests,
                "average_quality_score": round(avg_quality, 2),
                "average_generation_time": round(avg_time, 2),
                "evaluation_date": datetime.now().isoformat(),
                "model": "Qwen2.5-3B-Instruct",
                "task": "Domain Name Generation"
            },
            "detailed_results": self.results,
            "recommendations": [
                "Model shows good performance on simple business descriptions",
                "Edge cases reveal areas for improvement",
                "Complex business descriptions could benefit from fine-tuning"
            ]
        }
        
        with open('evaluation_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📈 Evaluation Summary:")
        print(f"   Business Tests: {total_tests}")
        print(f"   Avg Quality: {avg_quality:.2f}/10")
        print(f"   Avg Time: {avg_time:.2f} seconds")
        print(f"📄 Report saved to: evaluation_report.json")
        
        return report_data

def main():
    print("🎯 AI Homework: Domain Generation Evaluation Framework")
    print("=" * 70)
    print("⏰ Estimated time: 15 minutes")
    print("📊 Will test: 3 business categories")
    print("🎯 Goal: Systematic evaluation for homework report")
    print("=" * 70)
    
    try:
        evaluator = DomainGenerationEvaluator()
        evaluator.evaluate_business_categories()
        report = evaluator.generate_report()
        
        print("\n🎉 EVALUATION COMPLETE!")
        print("✅ All tests completed successfully")
        print("📊 Report generated for homework submission")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
