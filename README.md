# LoRA Fine-Tuning TinyLlama on Garden Questions
**Domain-specific fine-tuning of a 1.1B parameter LLM for horticultural Q&A**

Fine-tuning **TinyLlama-1.1B-Chat** on a custom gardening dataset using LoRA (Low-Rank Adaptation), with quantitative evaluation comparing the fine-tuned model against the base model baseline.

---

## Results

| Metric | Base TinyLlama | LoRA Fine-tuned | Improvement |
|--------|---------------|-----------------|-------------|
| BLEU | 0.0069 | 0.0254 | **+268%** |
| ROUGE-1 | 0.1780 | 0.3155 | **+77%** |
| ROUGE-L | 0.1195 | 0.2194 | **+84%** |
| METEOR | 0.2048 | 0.2054 | +0.3% |

The fine-tuned model provides concise, domain-specific answers rather than verbose generic responses. For example, when asked about tomato sunlight requirements, the base model gives a lengthy explanation while the fine-tuned model answers directly: *"Tomato plants require at least six hours of direct sunlight per day."*

---

## Dataset

**939 gardening Q&A pairs** (`dataset_plants_v5.jsonl`) covering:
- Plant nutrition and soil management
- Pest and disease prevention
- Specific vegetable growing advice
- Horticultural techniques

90/10 train/test split. Additional held-out test set of 30 examples for evaluation.

---

## Approach

**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Technique:** LoRA — only query and value projection weights adapted, keeping the base model frozen

**LoRA config:**
```
r=4 · alpha=16 · target_modules=[q_proj, v_proj] · dropout=0.05
```

**Training:**
```
epochs=3 · batch_size=2 · grad_accumulation=4 · lr=1e-4 · max_length=256
```

Optimised for **Mac Silicon (MPS)** with FP32 precision.

**Prompt format:**
```
<|user|>
{instruction}
<|assistant|>
{response}
<|endoftext|>
```

---

## Repository Structure

```
├── lora_gardening.py                  # Training script
├── evaluate_lora.py                   # Fine-tuned model evaluation
├── evaluate_without_lora.py           # Base model baseline evaluation
├── analysis.py                        # Dataset token length analysis
├── dataset_plants_v5.jsonl            # Training dataset (939 examples)
├── GardenWise_test_dataset_6.jsonl    # Test dataset (30 examples)
├── evaluation_results.json            # LoRA evaluation metrics
├── base_model_evaluation_results.json # Base model metrics
└── requirements.txt
```

---

## Running

```bash
pip install -r requirements.txt

# Train
python lora_gardening.py

# Evaluate fine-tuned vs base
python evaluate_lora.py
python evaluate_without_lora.py
```

---

## Stack

`transformers` · `peft` · `torch` (MPS) · `datasets` · `nltk` · `rouge-score`
