# LoRA fine-tuning for Mac Silicon with debugging
import os
import json
import torch
from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure environment
os.environ["WANDB_DISABLED"] = "true"

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data path
data_path = "dataset_plants_v5.jsonl"

# Load and preprocess dataset
formatted_data = []
with open(data_path) as file:
    for line in file:
        features = json.loads(line)
        # Combine input and output for causal LM
        full_text = f"<|user|>\n{features['instruction']}\n<|assistant|>\n{features['response']}\n<|endoftext|>"
        formatted_example = {
            "text": full_text
        }
        formatted_data.append(formatted_example)

# Create dataset
dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
print(f"Dataset size: {len(dataset)} examples")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with explicit device placement
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": device},
    torch_dtype=torch.float32  # Use FP32 explicitly
)

# Critical debugging step: verify model parameters require grad
print("Base model parameters requiring grad before LoRA:",
      sum(p.requires_grad for p in model.parameters()))

# Configure LoRA - simplified target modules
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Simplified for debugging
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Debug: Print model structure
print("Model structure before LoRA:")
for name, module in model.named_modules():
    if "q_proj" in name or "v_proj" in name:
        print(f"Found target module: {name}")

# Apply LoRA
model = get_peft_model(model, lora_config)

# Debug: Verify trainable parameters
trainable_params = 0
all_params = 0
for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)"
)

# Debug: Check if any parameters require grad
if trainable_params == 0:
    raise ValueError("No trainable parameters! LoRA setup failed.")


# Tokenize with labels for causal LM
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,  # Back to 256 for full training
        padding="max_length",
    )

    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Debug: Check tokenized data
print("Sample tokenized training example:")
print(tokenized_train[0])

# Use standard data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training args - simplified for debugging
training_args = TrainingArguments(
    output_dir="lora-tinyllama-garden",
    num_train_epochs=3,                   # Increase from 1 to 3 epochs
    per_device_train_batch_size=2,        # Try 2 if memory allows
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,                       # Evaluate less frequently
    save_strategy="steps",
    save_steps=100,                       # Save less frequently
    fp16=False,
    report_to="none",
    save_total_limit=2,                   # Keep last 2 checkpoints
    gradient_checkpointing=False,         # Keep disabled if it works
    # Critical for MPS
    optim="adamw_torch",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Debug: Verify forward pass works
print("\nTesting forward pass...")
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(device) for k, v in batch.items()}
with torch.no_grad():
    outputs = model(**batch)
print(f"Forward pass loss: {outputs.loss.item()}")

# Debug: Test backward pass with single batch
print("\nTesting backward pass...")
model.train()
outputs = model(**batch)
loss = outputs.loss
try:
    loss.backward()
    print("✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    raise

# Start training
print("\nStarting training...")
trainer.train()

# Save model
adapter_path = "lora-tinyllama-garden-final"
model.save_pretrained(adapter_path)
print(f"LoRA adapters saved to {adapter_path}")


# Test inference
def generate_response(instruction, max_length=200):
    prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test examples
test_questions = [
    "What are the ideal light conditions for growing peppers?",
    "What are the benefits of using mulch in the garden?"
]

print("\n===== SAMPLE OUTPUTS =====")
for question in test_questions:
    response = generate_response(question)
    print(f"Q: {question}")
    print(f"A: {response.split('<|assistant|>')[1]}")
    print("-" * 50)