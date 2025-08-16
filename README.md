# AI Domain Name Generator LLM

## Project Overview
AI Engineer Homework: Fine-tuned Large Language Model for Domain Name Generation

This project demonstrates systematic evaluation, edge case discovery, and iterative improvement of a fine-tuned LLM for generating creative domain name suggestions based on business descriptions.

##  Requirements Met
 Fine-tuned LLM for domain generation  
 Systematic evaluation framework with quality scoring  
 Edge case discovery and failure mode analysis  
 Iterative improvement approach  
 Safety guardrails implementation  
 Reproducible experiments  
 Technical documentation  
 Bonus: Production API deployment  

##  Key Results
- Quality Improvement: 6.3/10 → 7.7/10 (+21.1%)
- LoRA Efficiency: 0.12% trainable parameters
- Stable Fine-tuning: No numerical instability
- Memory Optimization: Apple Silicon MPS support

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
