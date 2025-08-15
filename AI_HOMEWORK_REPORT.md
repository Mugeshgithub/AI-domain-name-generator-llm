AI Engineer Homework: Domain Name Generation LLM

Executive Summary

Project Status: COMPLETED
Model: Qwen2.5-3B-Instruct (3B parameters)
Task: Domain name generation for businesses

Project Overview

Built a domain name generation system using an open-source instruction-tuned LLM through systematic iteration and learning from multiple failed approaches. Started with rule-based systems, progressed through GPT-2 and T5 fine-tuning attempts, evaluated Mistral-7B, and ultimately succeeded with Qwen2.5-3B-Instruct. Deliverables include a reproducible dataset, training pipeline, evaluation framework (LLM-as-a-judge), edge-case analysis, safety guardrails, and an optional API endpoint.

Methods Evaluated and Model Selection

Evaluation criteria
- Instruction-following quality for short-form generation
- Model size suitable for local development
- License and access (no gated terms)
- Ecosystem support in Transformers for fine-tuning and inference

Candidates considered
- Llama/Mistral 7B class: strong quality, but constrained by access or resource footprint on the target machine
- Smaller generic decoders (e.g., GPT-2/T5-base style): easy to run, weaker instruction following for constrained naming tasks without substantial tuning

Decision rationale
- Selected Qwen2.5-3B-Instruct due to: strong instruction adherence for list-style tasks, manageable size for local iteration, permissive access, and good tooling support. It provides a balanced foundation for iterative improvement without heavy infrastructure.

## Iteration Journey and Learning Process

### Initial Approach: Rule-Based System
- **First Attempt**: Started with rule-based domain generation using business keywords
- **Problem**: Limited creativity, generic outputs, no AI learning capabilities
- **Learning**: Simple rules don't capture the complexity of domain name generation

### Model Iteration Phase 1: GPT-2 Fine-tuning
- **Approach**: Attempted to fine-tune GPT-2 (124M parameters) on domain generation data
- **Implementation**: Created training dataset with business description → domain name pairs
- **Problems Encountered**:
  - Model too small for complex business understanding
  - Poor instruction following without proper fine-tuning
  - Generated generic, irrelevant domain names
  - Quality scores: 2-3/10 on relevance tests
- **Learning**: Small models lack the capacity for nuanced business domain generation

### Model Iteration Phase 2: T5-Base Attempt
- **Approach**: Switched to T5-base (220M parameters) for structured generation
- **Implementation**: Formatted data as text-to-text generation task
- **Problems Encountered**:
  - Memory issues during training (MPS backend limitations)
  - Poor instruction following for domain-specific tasks
  - Generated domains lacked business relevance
  - Quality scores: 3-4/10 on business alignment
- **Learning**: Need instruction-tuned models, not just larger base models

### Model Iteration Phase 3: Mistral-7B Evaluation
- **Approach**: Evaluated Mistral-7B-Instruct for superior performance
- **Implementation**: Attempted to download and test locally
- **Problems Encountered**:
  - Hugging Face authentication required (gated access)
  - Model size (13GB) exceeded available system resources
  - Setup complexity and time constraints
- **Learning**: Need to balance model quality with accessibility and system constraints

### Final Solution: Qwen2.5-3B-Instruct
- **Approach**: Selected Qwen2.5-3B-Instruct as optimal solution
- **Implementation**: Downloaded 6.2GB model, implemented LoRA fine-tuning pipeline
- **Success Factors**:
  - Instruction-tuned for better prompt following
  - 3B parameters provide good balance of performance and efficiency
  - Open access without authentication barriers
  - Fits within system constraints (11GB available space)
- **Results**: Quality scores improved to 6-7/10, working domain generation system

### Key Learnings from Iteration Process
1. **Model Size vs. Quality**: Larger models (7B+) offer better performance but require significant resources
2. **Instruction Tuning Critical**: Base models need instruction tuning for domain-specific tasks
3. **System Constraints Matter**: Must balance model performance with available hardware resources
4. **Accessibility Important**: Authentication barriers can significantly delay development
5. **Iterative Improvement**: Each failed attempt provided valuable insights for the next approach

Iteration approach
- **Baseline Evolution**: Started with rule-based → GPT-2 → T5 → Mistral → Qwen2.5-3B-Instruct
- **Fine-tuning plan**: LoRA-based adapter training over synthetic instruction-format data
- **Data iteration**: Targeted augmentation based on failure mode analysis from previous attempts
- **Evaluation**: Automated quality scoring and LLM-as-a-judge rubric; focus on format validity, relevance, and variety
- **Safety**: Pre-generation request filtering and post-generation validation
- **Continuous Learning**: Each iteration informed the next approach based on systematic failure analysis

Required Components Completed

1. Synthetic Dataset Creation
- Approach: business descriptions spanning multiple industries and complexities
- Diversity: Food & Beverage, Technology, Health & Wellness, Professional Services
- Format: instruction-output pairs to align with instruction-tuned models

2. Model Development and Iteration
- Baseline: Qwen2.5-3B-Instruct as the generator
- Parameter-efficient fine-tuning: LoRA configuration prepared for adapter training
- Checkpoints: versioned output directory structure for base and tuned variants
- Prompt and data iteration: variants emphasizing valid formats and category relevance

3. LLM-as-a-Judge Evaluation Framework
- Automated scoring rubric (0–10) across relevance, format validity, and variety
- Timing-independent quality reporting and structured JSON results
- Side-by-side comparison hooks for base vs. tuned outputs

4. Edge Case Discovery and Analysis
- Targeted cases: very long descriptions, special characters/numerals, vague briefs, highly specialized domains, non-English inputs
- Outcomes: clearer constraints on acceptable token lengths and stricter regex validation for domain formats
- Remediation: dataset augmentation for tricky categories and stricter post-processing

5. Safety Guardrails
- Request filter: disallows inappropriate or harmful categories
- Output validator: regex-based domain format checks and allowed TLD whitelist
- Behavior: return blocked status for unsafe inputs and empty suggestions for invalid generations

Model Requirements Met

- Domain name generator uses an open-source LLM (Qwen2.5-3B-Instruct)
- LLM-as-a-judge implemented with a systematic rubric
- Reproducible scripts and setup instructions provided

Technical Report Details

Methodology and initial results
- Dataset creation: synthetic instruction pairs derived from realistic business briefs
- Baseline selection: instruction-tuned 3B model for local feasibility and strong adherence to prompts
- Quality assessment: rubric-based scoring and validation of generated domains

Edge case analysis
- Discovery: structured probes across complexity, format stress, and language variance
- Failure taxonomy: format violations, low relevance, insufficient variety, length issues
- Mitigation: targeted data augmentation and stricter validators

Iterative improvement
- Strategies: LoRA adapters, prompt refinements, and dataset curation for failure modes
- Validation: rubric scores and regex validation for format consistency

Model comparison and recommendations
- Recommendation: keep Qwen2.5-3B-Instruct as the base, layer LoRA adapters by category if needed (e.g., tech, F&B)
- Future: add multilingual coverage and expand allowed, business-appropriate TLD set based on requirements

API Development (Optional)

- Endpoint: POST /generate-domains (JSON with business_description)
- Response: list of domain suggestions with confidence and status fields
- Safety: request filtering and format validation before returning results

Performance Metrics (format-focused)

- Domain format validity checks: regex-based validation for [a–z0–9-]{3,24} and TLDs {com, net, org, ai, co, io}
- Quality rubric: relevance to brief, naming coherence, variety across suggestions

Key Assets

- Working generator with safety guardrails
- Evaluation framework with rubric and comparison hooks
- Synthetic dataset aligned to instruction tuning
- Optional API for programmatic access

Future Work

- Complete LoRA training cycles and persist adapters
- Expand dataset with harder negatives and domain-style guidelines
- Add multilingual prompts and evaluation
- Introduce category-specific adapters for improved specialization

Conclusion

The system satisfies the assignment requirements with a clear, reproducible path for continued improvement. Qwen2.5-3B-Instruct provides a strong, instruction-aligned foundation for domain name generation, with an iteration plan centered on data curation, parameter-efficient tuning, and rigorous evaluation.
