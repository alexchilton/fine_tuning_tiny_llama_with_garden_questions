import json
import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the BASE model (no fine-tuning)
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Load tokenizer and BASE model (no LoRA)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map={"": device},
    torch_dtype=torch.float32
)
model.eval()
print("Loaded BASE model without fine-tuning")


# Function to read the jsonl file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


# Function to generate response from BASE model
def generate_base_response(instruction, max_new_tokens=256):
    prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    try:
        response = response.split("<|assistant|>\n")[1].split("<|endoftext|>")[0].strip()
    except IndexError:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[1].strip()
        else:
            response = response.split("<|user|>")[0].strip()

    return response


# Test data path
test_data_path = "GardenWise_test_dataset_6.jsonl"

# Define scoring objects and smoothing function
smoothing_function = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Initialize cumulative scores and counters
total_bleu, total_rouge1, total_rougeL, total_meteor_score, count = 0, 0, 0, 0, 0

# Load test data
test_data = read_jsonl(test_data_path)
print(f"Loaded {len(test_data)} test examples")

# Save individual results for analysis
base_results = []

# Iterate over the test data, generate responses with BASE model, and calculate scores
for i, entry in enumerate(test_data):
    print(f"Processing example {i + 1}/{len(test_data)}")

    # Extract instruction and expected response
    instruction = entry['instruction']
    expected_response = entry['response']

    # Generate a response using the BASE model
    base_response = generate_base_response(instruction)

    # Calculate BLEU score with smoothing
    bleu_score = sentence_bleu([expected_response.split()],
                               base_response.split(),
                               smoothing_function=smoothing_function)

    # Calculate ROUGE scores
    rouge_scores = rouge.score(expected_response, base_response)

    # Calculate METEOR score
    meteor = meteor_score([expected_response.split()], base_response.split())

    # Accumulate scores
    total_bleu += bleu_score
    total_rouge1 += rouge_scores['rouge1'].fmeasure
    total_rougeL += rouge_scores['rougeL'].fmeasure
    total_meteor_score += meteor
    count += 1

    # Save individual result
    base_results.append({
        'instruction': instruction,
        'expected': expected_response,
        'generated': base_response,
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'meteor': meteor
    })

    # Print progress every 10 examples
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} examples")

# Calculate average scores
average_bleu = total_bleu / count if count > 0 else 0
average_rouge1 = total_rouge1 / count if count > 0 else 0
average_rougeL = total_rougeL / count if count > 0 else 0
average_meteor_score = total_meteor_score / count if count > 0 else 0

# Print evaluation results
print("\n===== BASE MODEL EVALUATION RESULTS =====")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Average ROUGE-1 F1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-L F1 Score: {average_rougeL:.4f}")
print(f"Average METEOR Score: {average_meteor_score:.4f}")

# Save detailed results to file
with open('base_model_evaluation_results.json', 'w') as f:
    json.dump({
        'summary': {
            'bleu': average_bleu,
            'rouge1': average_rouge1,
            'rougeL': average_rougeL,
            'meteor': average_meteor_score,
            'count': count
        },
        'individual_results': base_results
    }, f, indent=2)

print(f"\nDetailed results saved to base_model_evaluation_results.json")

# Print some example comparisons
print("\n===== BASE MODEL EXAMPLE RESPONSES =====")
for i in range(min(3, len(base_results))):
    print(f"\nExample {i + 1}:")
    print(f"Instruction: {base_results[i]['instruction']}")
    print(f"Expected: {base_results[i]['expected'][:100]}...")
    print(f"BASE model: {base_results[i]['generated'][:100]}...")
    print(
        f"Scores: BLEU={base_results[i]['bleu']:.4f}, ROUGE-1={base_results[i]['rouge1']:.4f}, METEOR={base_results[i]['meteor']:.4f}")