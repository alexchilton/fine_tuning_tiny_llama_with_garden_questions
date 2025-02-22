# Fined tuned llama llm on Garden questions

I saw an interesting prject on kaggle - or rather my wife did! 

I took tiny llama and trained a fine tuned model on 993 questions and answers, added metrics to check the results.
Nothing too fancy but it worked 

The implementation is using a lora matrix of rank 4. Trained for 3 epochs

Comparing results, we can see your LoRA fine-tuning definitely improved performance over the base model:

| Metric | Base TinyLlama | LoRA Fine-tuned | Improvement |
|--------|---------------|----------------|-------------|
| BLEU   | 0.0069        | 0.0254         | +268%       |
| ROUGE-1| 0.1780        | 0.3155         | +77%        |
| ROUGE-L| 0.1195        | 0.2194         | +84%        |
| METEOR | 0.2048        | 0.2054         | +0.3%       |

There are clear improvements in most metrics:
- BLEU score improved dramatically (nearly 4x better)
- ROUGE-1 improved by 77% 
- ROUGE-L improved by 84%
- METEOR stayed essentially the same

Looking at the examples, the fine-tuned model gives more specific gardening advice while the base model tends to provide more generic information. For instance, in the "easiest vegetables" question, the fine-tuned model immediately lists specific vegetables, while the base model gives a general definition.

The LoRA fine-tuning has successfully specialized the model for gardening advice, particularly improving:
1. Lexical overlap with expert answers (ROUGE/BLEU)
2. Structure of responses for gardening questions
3. Domain-specific vocabulary usage

The METEOR score staying nearly the same suggests that both models maintain similar semantic understanding, but your fine-tuned model produces text that more closely matches expert gardening advice in terms of specific content and terminology.
