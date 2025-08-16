# Local Setup Guide for FamilyWall AI Engineer Homework

## Important: This project requires large model files that are NOT on GitHub

Due to GitHub's file size limitations, the following critical components are **NOT included** in the repository but are required for the project to function:

## 🚨 Required Local Files (NOT on GitHub)

### 1. Base Model: `qwen2.5-3b-instruct/`
- **Size**: ~5.8GB
- **Contents**: Qwen2.5-3B-Instruct base model files
- **Location**: `./qwen2.5-3b-instruct/`
- **How to get**: Run `python scripts/download_qwen.py`

### 2. Fine-tuned Model: `fine_tuned_model_stable_backup/`
- **Size**: ~14MB
- **Contents**: LoRA fine-tuned model adapters
- **Location**: `./fine_tuned_model_stable_backup/`
- **Status**: Already present in your local folder

## 📁 What IS on GitHub

- ✅ Complete source code and scripts
- ✅ Training dataset (`training_dataset_fixed.json`)
- ✅ Evaluation reports (`evaluation_report.json`, `model_comparison_report.json`)
- ✅ Technical documentation (`TECHNICAL_REPORT.md`, `README.md`)
- ✅ API implementation (`api/` folder)
- ✅ Web application (`app/` folder)

## 🔧 Local Setup Steps

1. **Clone the repository** (you already have this)
2. **Download the base model**:
   ```bash
   cd familywall-ai-homework
   python scripts/download_qwen.py
   ```
3. **Verify models are present**:
   ```bash
   ls -lh qwen2.5-3b-instruct/
   ls -lh fine_tuned_model_stable_backup/
   ```
4. **Test the evaluation**:
   ```bash
   python scripts/evaluate_improvements.py
   ```
5. **Run the API**:
   ```bash
   cd api && python main.py
   ```

## 🚫 Why Models Aren't on GitHub

- **GitHub file size limit**: 100MB per file
- **Base model**: 3.7GB + 2.1GB = 5.8GB total
- **Fine-tuned model**: 14MB (could fit, but requires base model)
- **Git LFS**: Not configured for this project

## 📋 For Interview/Demonstration

When presenting this project:
1. **Show the code** on GitHub
2. **Demonstrate locally** with the models loaded
3. **Explain the architecture** and improvements
4. **Highlight the 21.1% quality improvement** achieved

## 🔍 Troubleshooting

If you get "model not found" errors:
1. Check if `qwen2.5-3b-instruct/` exists and contains model files
2. Verify `fine_tuned_model_stable_backup/` is present
3. Ensure you're running scripts from the correct directory

## 📊 Project Status

- **Code**: ✅ Complete and on GitHub
- **Documentation**: ✅ Complete and on GitHub  
- **Base Model**: ❌ Needs local download (~5.8GB)
- **Fine-tuned Model**: ✅ Present locally (~14MB)
- **Evaluation**: ✅ Ready to run locally
- **API**: ✅ Ready to run locally

This setup ensures you have a complete, working project while keeping the repository lightweight and GitHub-compatible.
