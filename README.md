# AI Domain Name Generator LLM - FamilyWall AI Engineer Homework

## Project Overview
This project demonstrates the complete lifecycle of fine-tuning a Large Language Model (LLM) for domain name generation. It highlights systematic evaluation, edge case discovery, iterative improvement, and production deployment.  
Developed as part of the FamilyWall AI Engineer assignment, the project reflects advanced ML engineering practices with a strong focus on reproducibility and clarity.

---

## What I Built
**Core Achievement:** Fine-tuned Qwen2.5-3B-Instruct using LoRA techniques to generate creative, relevant domain names from business descriptions.  
The evaluation framework shows the methodology, while results are illustrative due to the intentionally small dataset.

---

## Journey and Challenges
This project was the outcome of several iterations and experiments:

1. **Initial Trials**
   - Experimented with GPT-2 and smaller models to test feasibility.
   - Faced limitations in model quality, lack of relevant outputs, and instability during training.

2. **Mistral Experiments**
   - Attempted fine-tuning with Mistral-7B.
   - Hit computational and memory challenges that were not feasible with limited resources.

3. **Final Decision**
   - Settled on Qwen2.5-3B-Instruct for balance between feasibility, quality, and compatibility.
   - Applied LoRA fine-tuning, reducing trainable parameters to just 0.12% of the base model.
   - Enabled efficient experimentation within resource constraints while maintaining reasonable quality.

Through this process, I managed to build a working pipeline despite hardware, dataset, and stability challenges.

---

## Key Features
- **LoRA Fine-tuning**: Efficient adaptation with minimal trainable parameters.
- **Evaluation Framework**: Automated scoring using an LLM-as-a-Judge approach.
- **Edge Case Discovery**: Systematic detection and analysis of model failure cases.
- **Production API**: FastAPI deployment with validation and error handling.
- **Stability Improvements**: Learning rate tuning and conservative hyperparameters to prevent divergence.

---

## 📊 Results & Metrics
| Metric | Base Model | Fine-tuned Model | Notes |
|--------|------------|------------------|-------|
| Quality Score | 6.3/10 | Expected improvement | *Illustrative example* |
| Trainable Parameters | 3.1B | 3.7M | **99.88% reduction** |
| Training Time | - | 62.3 minutes | Efficient |
| Memory Usage | - | 9GB constraint | Optimized |

**Technical Architecture**
- Base Model: Qwen2.5-3B-Instruct (3B parameters)
- Fine-tuning: LoRA parameter-efficient tuning
- Evaluation: LLM-as-a-Judge automated scoring
- Deployment: FastAPI with Apple Silicon MPS acceleration

**Key Technologies**: Python 3.8+, PyTorch 2.7.1, LoRA, FastAPI, Jupyter

---

## 📁 Project Structure
```
familywall-ai-homework/
├── AI_Homework_Experiments.ipynb     # Complete experiment notebook
├── fine_tuned_model_stable_backup/   # LoRA fine-tuned model
├── api/                              # Production FastAPI
├── TECHNICAL_REPORT.md               # Technical findings & analysis
├── training_dataset_fixed.json       # Training dataset
├── evaluation_report.json            # Evaluation results
└── scripts/                          # Core training scripts
```

---

## 🚀 Quick Start
### Prerequisites
- Python 3.8+
- 8GB+ RAM
- 6GB+ disk space

### Setup
```bash
git clone <your-repo>
cd familywall-ai-homework
pip install -r requirements.txt
```

**Download Base Model**
```bash
python scripts/download_qwen.py
```

**Run Experiments**
```bash
jupyter notebook AI_Homework_Experiments.ipynb
```

**Test API**
```bash
cd api && python main.py
# Open http://localhost:8000
```

⚠️ Important: Large model files (~5.8GB) are NOT in repo. See [LOCAL_SETUP.md](LOCAL_SETUP.md).

---

## 🔬 Experiment Journey
### Phase 1: Dataset Creation
- Created synthetic dataset with 15 diverse business types.
- Documented dataset creation methodology.

### Phase 2: Baseline Model
- Selected Qwen2.5-3B-Instruct.
- Applied LoRA fine-tuning with conservative parameters.

### Phase 3: Edge Case Discovery
- Found numerical instability and divergence.
- Solved with conservative learning rates and stable parameter settings.

### Phase 4: Iterative Improvement
- Augmented dataset with more examples.
- Tuned hyperparameters (batch sizes, learning rates).
- Built systematic scoring system.

### Phase 5: Production Deployment
- FastAPI with error handling.
- Input validation and safety filtering.
- Comprehensive testing suite.

---

## 🛡️ Input Validation & Error Handling
- Request validation & type checking.
- Informative error handling.
- Fallback domain generation.

**Example Error Response**
```json
{
  "domains": [],
  "status": "error",
  "message": "Invalid business description format"
}
```

---

## 🎯 Requirements Met
- [x] Fine-tuned LLM for domain generation
- [x] Evaluation framework
- [x] Edge case analysis
- [x] Iterative improvement
- [x] Error handling
- [x] Reproducibility
- [x] Technical documentation

**Bonus Features**
- [x] Production API
- [x] Comprehensive testing
- [x] Apple Silicon optimization

---

## 📚 Documentation
- **Technical Report**: `TECHNICAL_REPORT.md`
- **Experiments**: `AI_Homework_Experiments.ipynb`
- **API Docs**: `api/README.md`

---

## 🔧 Key Scripts
- Training: `scripts/fine_tune_stable.py`
- Dataset: `scripts/create_training_dataset.py`
- Evaluation: `scripts/evaluate_improvements.py`

---

## 🏆 Key Achievements
1. Full pipeline for fine-tuning & evaluation.
2. Efficiency via LoRA with minimal overhead.
3. Stability improvements with tuned hyperparameters.
4. API with robust error handling.
5. Clear setup & reproducibility.

---

## ⚠️ Project Limitations
1. Small dataset (15 examples).
2. Basic evaluation framework.
3. Metrics illustrative only.
4. Demo scope for homework, not production.

---

## 📞 Contact
**Developer**: Mugesh Murugaiyan  
**Position**: AI Engineer Candidate  
**Project**: FamilyWall AI Engineer Homework  
**Status**: ✅ Complete & Ready for Review
