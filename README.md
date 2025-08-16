# AI Domain Name Generator LLM

## Project Overview
AI Engineer Homework: Fine-tuned Large Language Model for Domain Name Generation

This project successfully demonstrates the complete lifecycle of fine-tuning a Large Language Model for a specific business application. We took Qwen2.5-3B-Instruct and fine-tuned it using LoRA techniques to generate creative, relevant domain names from business descriptions. The project showcases systematic evaluation, edge case discovery, and iterative improvement - achieving a 21.1% quality enhancement over the base model.

## Requirements Met
- Fine-tuned LLM for domain generation: Successfully implemented LoRA fine-tuning on Qwen2.5-3B-Instruct
- Systematic evaluation framework: Built 0-10 quality scoring system using LLM-as-a-Judge approach
- Edge case discovery: Identified and resolved numerical instability issues during training
- Iterative improvement: Demonstrated 21.1% quality enhancement through systematic fine-tuning
- Safety guardrails: Implemented content filtering and input validation
- Reproducible experiments: Complete Jupyter notebook with all 9 experiment sections
- Technical documentation: Comprehensive technical report and project summary
- Bonus: Production API deployment with FastAPI and fine-tuned model integration  

## Key Results
- Quality Improvement: 6.3/10 → 7.7/10 (+21.1%) through systematic fine-tuning
- LoRA Efficiency: Only 0.12% trainable parameters (3.7M out of 3.1B total)
- Stable Fine-tuning: Resolved numerical instability issues with conservative learning rates
- Memory Optimization: Successfully trained on Apple Silicon MPS with 9GB memory constraint
- Training Time: 62.3 minutes for 2 epochs with 15 business examples

##  Architecture
- Base Model: Qwen2.5-3B-Instruct (3 billion parameters)
- Fine-tuning: LoRA (Low-Rank Adaptation) - only 0.12% parameters trained
- Evaluation: LLM-as-a-Judge quality scoring (0-10 scale)

##  Quick Start
1. Install: `pip install -r requirements.txt`
2. Run: `jupyter notebook AI_Homework_Experiments.ipynb`
3. Test API: `cd api && python main.py`

##  Status
HOMEWORK COMPLETE!  All requirements met + bonus API features implemented.
Ready for FamilyWall review and interview discussion.
