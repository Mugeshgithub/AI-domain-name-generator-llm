# AI Domain Name Generator LLM - FamilyWall AI Engineer Homework

## 🎯 Project Overview

This project demonstrates the complete lifecycle of fine-tuning a Large Language Model for domain name generation, showcasing systematic evaluation, edge case discovery, and iterative improvement. Built for the FamilyWall AI Engineer position, it demonstrates advanced ML engineering skills with a focus on reproducible research and production deployment.

## 🚀 What We Built

**Core Achievement**: Fine-tuned Qwen2.5-3B-Instruct using LoRA techniques to generate creative, relevant domain names from business descriptions, achieving a **21.1% quality improvement** over the base model.

### Key Features
- **LoRA Fine-tuning**: Efficient adaptation using only 0.12% trainable parameters
- **LLM-as-a-Judge Evaluation**: Automated quality scoring system (0-10 scale)
- **Edge Case Discovery**: Systematic identification and resolution of model failures
- **Production API**: FastAPI deployment with comprehensive error handling
- **Safety Guardrails**: Content filtering for inappropriate requests

## 📊 Results & Metrics

| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| Quality Score | 6.3/10 | 7.7/10 | **+21.1%** |
| Trainable Parameters | 3.1B | 3.7M | **99.88% reduction** |
| Training Time | - | 62.3 minutes | Efficient |
| Memory Usage | - | 9GB constraint | Optimized |

## 🏗️ Technical Architecture

### Model Stack
- **Base Model**: Qwen2.5-3B-Instruct (3 billion parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficiency
- **Evaluation**: LLM-as-a-Judge using quality scoring framework
- **Deployment**: FastAPI with Apple Silicon MPS acceleration

### Key Technologies
- **Python 3.8+** with PyTorch 2.7.1
- **LoRA** for parameter-efficient fine-tuning
- **FastAPI** for production API deployment
- **Jupyter Notebooks** for reproducible experiments

## 📁 Project Structure

```
familywall-ai-homework/
├── 📊 AI_Homework_Experiments.ipynb     # Complete experiment notebook
├── 🤖 fine_tuned_model_stable/          # LoRA fine-tuned model
├── 🚀 api/                              # Production FastAPI
├── 📝 TECHNICAL_REPORT.md               # Technical findings & analysis
├── 📊 training_dataset_fixed.json       # Training dataset
├── 📈 evaluation_report.json            # Evaluation results
└── 🔧 scripts/                          # Core training scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM
- 6GB+ disk space

### Setup
1. **Clone & Install**
   ```bash
   git clone <your-repo>
   cd familywall-ai-homework
   pip install -r requirements.txt
   ```

2. **Download Base Model**
   ```bash
   python scripts/download_qwen.py
   ```

3. **Run Experiments**
   ```bash
   jupyter notebook AI_Homework_Experiments.ipynb
   ```

4. **Test API**
   ```bash
   cd api && python main.py
   # Open http://localhost:8000
   ```

## 🔬 Experiment Journey

### Phase 1: Dataset Creation
- Created synthetic dataset with 15 diverse business types
- Implemented systematic data generation methodology
- Documented dataset creation approach

### Phase 2: Baseline Model
- Selected Qwen2.5-3B-Instruct as base model
- Implemented LoRA fine-tuning with conservative parameters
- Achieved initial working model

### Phase 3: Edge Case Discovery
- **Identified Issues**: Numerical instability, training divergence
- **Root Causes**: High learning rates, unstable LoRA config
- **Solutions**: Conservative learning rates, stable parameter settings

### Phase 4: Iterative Improvement
- **Dataset Augmentation**: Enhanced training examples
- **Hyperparameter Optimization**: Fine-tuned learning rates and batch sizes
- **Evaluation Framework**: Built systematic quality scoring system

### Phase 5: Production Deployment
- **API Development**: FastAPI with comprehensive error handling
- **Safety Implementation**: Content filtering and input validation
- **Testing**: Comprehensive test suite and validation

## 📈 Evaluation Framework

### LLM-as-a-Judge Implementation
- **Quality Scoring**: 0-10 scale for domain name relevance
- **Automated Evaluation**: Systematic assessment of model outputs
- **Statistical Analysis**: Before/after comparison with significance testing

### Edge Case Analysis
- **Failure Taxonomy**: Categorized different types of model failures
- **Frequency Analysis**: Quantified occurrence of each failure type
- **Improvement Tracking**: Measured progress in handling edge cases

## 🛡️ Safety & Guardrails

### Content Filtering
- **Inappropriate Content Detection**: Automatic blocking of harmful requests
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Graceful degradation with informative messages

### Example Safety Response
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## 🎯 Requirements Met

### ✅ Core Requirements
- [x] Fine-tuned LLM for domain generation
- [x] Systematic evaluation framework
- [x] Edge case discovery and analysis
- [x] Iterative improvement approach
- [x] Safety guardrails implementation
- [x] Reproducible experiments
- [x] Technical documentation

### 🚀 Bonus Features
- [x] Production API deployment
- [x] Comprehensive testing suite
- [x] Apple Silicon optimization

## 📚 Documentation

- **Technical Report**: `TECHNICAL_REPORT.md` - Comprehensive technical analysis
- **Experiments**: `AI_Homework_Experiments.ipynb` - Complete experiment notebook
- **API Docs**: `api/README.md` - API usage and testing

## 🔧 Key Scripts

- **Training**: `scripts/fine_tune_stable.py` - Main LoRA fine-tuning script
- **Dataset**: `scripts/create_training_dataset.py` - Dataset generation
- **Evaluation**: `scripts/evaluate_improvements.py` - Evaluation framework

## 🏆 Key Achievements

1. **Quality Improvement**: 21.1% enhancement over base model
2. **Efficiency**: LoRA fine-tuning with minimal parameter overhead
3. **Stability**: Resolved numerical instability issues
4. **Production Ready**: Complete API with error handling
5. **Reproducibility**: Clear setup instructions and version tracking

## 🎯 Interview Discussion Points

Be ready to discuss:
- **LoRA vs Full Fine-tuning**: Why we chose LoRA and the trade-offs
- **Edge Case Discovery**: How we systematically identified model failures
- **Evaluation Framework**: LLM-as-a-Judge implementation details
- **Safety Implementation**: Content filtering approach and testing
- **Production Considerations**: API design decisions and error handling

## 📞 Contact

**Developer**: Mugesh Murugaiyan  
**Position**: AI Engineer Candidate  
**Project**: FamilyWall AI Engineer Homework  
**Status**: ✅ Complete & Ready for Review

---

**This project demonstrates advanced ML engineering skills including systematic evaluation, edge case discovery, iterative improvement, and production deployment - all key requirements for the FamilyWall AI Engineer position.**
