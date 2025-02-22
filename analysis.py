import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from huggingface_hub import login
login(token="hftoken")


# Path to dataset
data_path = "dataset_plants_v5.jsonl"

# Load the TinyLlama tokenizer (same as we'll use for training)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Read and analyze the dataset
questions_lengths = []
answers_lengths = []
total_lengths = []
formatted_lengths = []

# Read the data
with open(data_path) as file:
    for line in file:
        features = json.loads(line)

        # Get token counts
        question_tokens = len(tokenizer.encode(features['instruction']))
        answer_tokens = len(tokenizer.encode(features['response']))
        total_tokens = question_tokens + answer_tokens

        # Calculate formatted example length (as it would appear in training)
        formatted_text = f"<|user|>\n{features['instruction']}\n<|assistant|>\n{features['response']}\n<|endoftext|>"
        formatted_tokens = len(tokenizer.encode(formatted_text))

        # Store lengths
        questions_lengths.append(question_tokens)
        answers_lengths.append(answer_tokens)
        total_lengths.append(total_tokens)
        formatted_lengths.append(formatted_tokens)

# Convert to numpy arrays
questions_lengths = np.array(questions_lengths)
answers_lengths = np.array(answers_lengths)
total_lengths = np.array(total_lengths)
formatted_lengths = np.array(formatted_lengths)

# Create statistics summary
length_stats = pd.DataFrame({
    'Statistic': ['Min', 'Max', 'Mean', 'Median', '90th percentile', '95th percentile', '99th percentile'],
    'Question Tokens': [
        np.min(questions_lengths),
        np.max(questions_lengths),
        np.mean(questions_lengths),
        np.median(questions_lengths),
        np.percentile(questions_lengths, 90),
        np.percentile(questions_lengths, 95),
        np.percentile(questions_lengths, 99)
    ],
    'Answer Tokens': [
        np.min(answers_lengths),
        np.max(answers_lengths),
        np.mean(answers_lengths),
        np.median(answers_lengths),
        np.percentile(answers_lengths, 90),
        np.percentile(answers_lengths, 95),
        np.percentile(answers_lengths, 99)
    ],
    'Total Tokens': [
        np.min(total_lengths),
        np.max(total_lengths),
        np.mean(total_lengths),
        np.median(total_lengths),
        np.percentile(total_lengths, 90),
        np.percentile(total_lengths, 95),
        np.percentile(total_lengths, 99)
    ],
    'Formatted Tokens': [
        np.min(formatted_lengths),
        np.max(formatted_lengths),
        np.mean(formatted_lengths),
        np.median(formatted_lengths),
        np.percentile(formatted_lengths, 90),
        np.percentile(formatted_lengths, 95),
        np.percentile(formatted_lengths, 99)
    ]
})

# Calculate percentages exceeding various thresholds
thresholds = [128, 256, 384, 512, 768, 1024]
threshold_data = []

for threshold in thresholds:
    pct_questions = (questions_lengths > threshold).mean() * 100
    pct_answers = (answers_lengths > threshold).mean() * 100
    pct_total = (total_lengths > threshold).mean() * 100
    pct_formatted = (formatted_lengths > threshold).mean() * 100

    threshold_data.append([
        threshold,
        pct_questions,
        pct_answers,
        pct_total,
        pct_formatted
    ])

threshold_df = pd.DataFrame(
    threshold_data,
    columns=['Threshold', 'Questions %', 'Answers %', 'Total %', 'Formatted %']
)

# Display statistics
print("=== Token Length Statistics ===")
print(length_stats.to_string(index=False))
print("\n=== Percentage Exceeding Thresholds ===")
print(threshold_df.to_string(index=False))

# Create histogram plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(questions_lengths, bins=30, alpha=0.7, color='blue')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.7)
plt.title('Question Token Lengths')
plt.xlabel('Tokens')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
plt.hist(answers_lengths, bins=30, alpha=0.7, color='green')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.7)
plt.title('Answer Token Lengths')
plt.xlabel('Tokens')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
plt.hist(total_lengths, bins=30, alpha=0.7, color='purple')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.7)
plt.title('Total Token Lengths (Q+A)')
plt.xlabel('Tokens')
plt.ylabel('Count')

plt.subplot(2, 2, 4)
plt.hist(formatted_lengths, bins=30, alpha=0.7, color='orange')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.7)
plt.title('Formatted Example Lengths')
plt.xlabel('Tokens')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('token_length_analysis.png')
plt.show()

print("\nAnalysis complete. Visualization saved as 'token_length_analysis.png'")