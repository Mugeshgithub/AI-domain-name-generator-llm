#!/usr/bin/env python3
"""
Stable Fine-tuning Implementation
Fixed version with conservative parameters to avoid numerical instability
"""

import torch
import json
import time
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc

def stable_fine_tune():
    print("🔧 STABLE FINE-TUNING - FIXING NUMERICAL INSTABILITY")
    print("=" * 60)
    
    start_total = time.time()
    
    # System check
    print("📊 SYSTEM VALIDATION:")
    print(f"   ✅ Apple Silicon GPU (MPS): {torch.backends.mps.is_available()}")
    print(f"   ✅ PyTorch version: {torch.__version__}")
    
    # Load training dataset
    print("\n📥 LOADING TRAINING DATASET:")
    with open("training_dataset_fixed.json", "r") as f:
        training_data = json.load(f)
    
    print(f"   ✅ Training examples: {len(training_data)}")
    
    # Load tokenizer
    print("\n🔧 LOADING TOKENIZER:")
    tokenizer = AutoTokenizer.from_pretrained('./qwen2.5-3b-instruct')
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")
    
    # Load base model
    print("\n🤖 LOADING BASE MODEL:")
    start_load = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        './qwen2.5-3b-instruct',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("   ✅ Model deployed to Apple Silicon GPU (MPS)")
    
    load_time = time.time() - start_load
    print(f"   ✅ Model loaded in {load_time:.1f} seconds")
    
    # STABLE LoRA configuration (conservative settings)
    print("\n🔧 CONFIGURING STABLE LoRA:")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,           # Reduced from 16 (more stable)
        lora_alpha=16, # Reduced from 64 (more stable)
        lora_dropout=0.05, # Reduced from 0.1 (more stable)
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Core attention only
        bias="none",
        use_rslora=False, # Disabled advanced features for stability
    )
    
    model = get_peft_model(model, lora_config)
    print("   ✅ Stable LoRA configured")
    model.print_trainable_parameters()
    
    # Prepare dataset with consistent formatting
    print("\n📊 PREPARING DATASET:")
    
    def format_stable_prompt(example):
        """Consistent, simple prompt formatting"""
        system_prompt = "You are an AI assistant that generates domain names for businesses."
        user_prompt = example['instruction']
        assistant_response = example['output']
        
        # Simple, consistent format
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>"
        return formatted_prompt
    
    # Create dataset
    formatted_data = [format_stable_prompt(ex) for ex in training_data]
    dataset = Dataset.from_dict({"text": formatted_data})
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=256, # Reduced from 512 for stability
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("   ✅ Dataset prepared")
    
    # STABLE training arguments (conservative settings)
    print("\n⚙️  CONFIGURING STABLE TRAINING:")
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model_stable",
        num_train_epochs=2,  # Reduced from 5 (more stable)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced from 8
        warmup_steps=10,  # Reduced from 20
        logging_steps=1,
        save_steps=10,  # More frequent saving
        save_strategy="steps",
        learning_rate=1e-4,  # Reduced from 3e-4 (more stable)
        weight_decay=0.01,
        max_grad_norm=0.5,  # Reduced from 1.0 (more stable)
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
        save_total_limit=2,
        # Add stability measures
        dataloader_num_workers=0,
        group_by_length=False,
        length_column_name="length",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("   ✅ Stable training configuration complete")
    
    # Training execution
    print("\n🎯 EXECUTING STABLE FINE-TUNING:")
    print("   Training for 2 epochs with conservative parameters...")
    
    start_training = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_training
        print(f"   ✅ Stable fine-tuning completed in {training_time:.1f} seconds")
        
        # Save model
        print("\n💾 SAVING STABLE MODEL:")
        trainer.save_model()
        print("   ✅ Stable model saved to ./fine_tuned_model_stable")
        
        # Test the model immediately
        print("\n🧪 TESTING STABLE MODEL:")
        test_stable_model()
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print("   This indicates a deeper issue with the setup")
    
    # Cleanup
    print("\n🧹 CLEANUP:")
    del model, trainer
    gc.collect()
    
    # Summary
    total_time = time.time() - start_total
    print(f"\n🏆 STABLE FINE-TUNING COMPLETE!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Training examples: {len(training_data)}")
    print(f"   Model saved: ./fine_tuned_model_stable")

def test_stable_model():
    """Quick test of the stable model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print("   Loading stable model for testing...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model_stable')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            './qwen2.5-3b-instruct',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load LoRA
        model = PeftModel.from_pretrained(model, './fine_tuned_model_stable')
        
        if torch.backends.mps.is_available():
            model = model.to("mps")
        
        # Quick test
        prompt = "<|im_start|>system\nYou are an AI assistant that generates domain names for businesses.<|im_end|>\n<|im_start|>user\nGenerate 3 domain names for: coffee shop<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Test successful! Generated: {response[-100:]}...")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        print("   The model still has issues")

if __name__ == "__main__":
    stable_fine_tune()
