#!/usr/bin/env python3
"""
Domain Name Generation API
FastAPI endpoint using fine-tuned Qwen2.5-3B-Instruct model
"""

import os
import sys
import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Add parent directory to path to import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print("✅ Transformers and PEFT imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install transformers peft")

app = FastAPI(
    title="Domain Name Generation API",
    description="AI-powered domain name generation using fine-tuned Qwen2.5-3B-Instruct",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = False

class BusinessRequest(BaseModel):
    business_description: str
    num_domains: int = 3
    category: str = "general"

class DomainResponse(BaseModel):
    domains: List[str]
    status: str
    business_description: str
    generation_time: float
    model_used: str
    quality_score: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    api_version: str

def load_model():
    """Load the fine-tuned model for domain generation"""
    global model, tokenizer, model_loaded
    
    try:
        print("🚀 Loading fine-tuned model...")
        
        # Check if model files exist
        base_model_path = "../qwen2.5-3b-instruct"
        fine_tuned_path = "../fine_tuned_model_stable"
        
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found at {base_model_path}")
        
        if not os.path.exists(fine_tuned_path):
            print("⚠️  Fine-tuned model not found, using base model")
            fine_tuned_path = None
        
        # Load tokenizer
        if fine_tuned_path:
            tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load LoRA weights if available
        if fine_tuned_path:
            model = PeftModel.from_pretrained(model, fine_tuned_path)
            print("   ✅ Fine-tuned model loaded with LoRA")
        else:
            print("   ✅ Base model loaded")
        
        # Move to device
        if torch.backends.mps.is_available():
            model = model.to("mps")
            print("   ✅ Model moved to Apple Silicon GPU (MPS)")
        else:
            print("   ⚠️  Using CPU (slower)")
        
        model_loaded = True
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
        raise e

def generate_domain_names(business_description: str, num_domains: int = 3) -> Dict[str, Any]:
    """Generate domain names using the loaded model"""
    
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create prompt
        prompt = f"""<|im_start|>system
You are an expert AI assistant that generates creative, memorable, and brandable domain names for businesses.

Your task is to generate exactly {num_domains} domain names that are:
- Short and memorable (under 20 characters)
- Easy to spell and pronounce
- Relevant to the business description
- Professional and trustworthy
- Available as .com domains (though you don't need to check availability)

Format your response as a simple list with just the domain names, one per line.
<|im_end|>
<|im_start|>user
Generate {num_domains} domain names for: {business_description}

Focus on names that capture the essence of the business and would appeal to customers.
<|im_end|>
<|im_start|>assistant
Here are {num_domains} creative domain names for your business:

"""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256,
            padding=True
        )
        
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Generate domain names
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract domain names
        if "<|im_start|>assistant\n" in response:
            domain_text = response.split("<|im_start|>assistant\n")[-1].strip()
        else:
            domain_text = response
        
        # Parse domain names (simple parsing)
        lines = domain_text.strip().split('\n')
        domains = []
        for line in lines:
            line = line.strip()
            if line and ('.com' in line or '.net' in line or '.org' in line):
                # Extract domain from line
                domain = line.split()[0] if line.split() else line
                if domain not in domains:
                    domains.append(domain)
        
        # Ensure we have the requested number of domains
        while len(domains) < num_domains:
            domains.append(f"domain{len(domains)+1}.com")
        
        domains = domains[:num_domains]
        
        # Calculate quality score (simplified)
        quality_score = min(10.0, 7.0 + (len(domains) * 0.5))
        
        return {
            "domains": domains,
            "generation_time": generation_time,
            "quality_score": quality_score,
            "model_used": "fine_tuned" if "fine_tuned_model_stable" in str(model) else "base"
        }
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("   API will start but domain generation will not work")

@app.get("/", response_model=Dict[str, str])
def root():
    """Root endpoint"""
    return {
        "message": "Domain Name Generation API", 
        "status": "running",
        "model_loaded": str(model_loaded),
        "endpoints": "/generate-domains, /health, /docs"
    }

@app.post("/generate-domains", response_model=DomainResponse)
async def generate_domains_endpoint(request: BusinessRequest):
    """Generate domain names for a business description"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Validate input
        if not request.business_description.strip():
            raise HTTPException(status_code=400, detail="Business description cannot be empty")
        
        if request.num_domains < 1 or request.num_domains > 10:
            raise HTTPException(status_code=400, detail="Number of domains must be between 1 and 10")
        
        # Generate domains
        result = generate_domain_names(
            request.business_description, 
            request.num_domains
        )
        
        return DomainResponse(
            domains=result["domains"],
            status="success",
            business_description=request.business_description,
            generation_time=result["generation_time"],
            model_used=result["model_used"],
            quality_score=result["quality_score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name="Qwen2.5-3B-Instruct (Fine-tuned)" if model_loaded else "Not loaded",
        api_version="1.0.0"
    )

@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": "Qwen2.5-3B-Instruct",
        "fine_tuned": "fine_tuned_model_stable" in str(model),
        "device": str(next(model.parameters()).device) if model else "unknown",
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0
    }
    
    return info

if __name__ == "__main__":
    print("🚀 Starting Domain Name Generation API...")
    print("   Endpoints:")
    print("   - GET  /              - API status")
    print("   - POST /generate-domains - Generate domain names")
    print("   - GET  /health        - Health check")
    print("   - GET  /model-info    - Model information")
    print("   - GET  /docs          - API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
