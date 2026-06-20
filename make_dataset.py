import json
import os

CONVERSATIONS = [
    {"q": "What is LoRA?", "a": "LoRA fine-tunes LLMs by learning small matrices A x B instead of updating all weights."},
    {"q": "How does fine-tuning work?", "a": "Fine-tuning continues training a pretrained model on specific data. LoRA makes this efficient."},
    {"q": "What is the transformer?", "a": "Transformers use self-attention. Key parts: multi-head attention and feed-forward networks."},
    {"q": "What is gradient checkpointing?", "a": "Trades compute for memory by not storing intermediate activations. Cuts GPU memory ~50%."},
    {"q": "What is mixed precision?", "a": "Uses FP16/BF16 for compute, FP32 for weights. Gives 2-3x speedup on GPU Tensor Cores."},
    {"q": "What is catastrophic forgetting?", "a": "When a model loses learned knowledge. LoRA and low LRs help prevent this."},
    {"q": "How does attention work?", "a": "Q,K,V projections. Attention=softmax(Q*K^T/sqrt(d))*V. GPT-2 uses causal masking."},
    {"q": "What does loss measure?", "a": "Cross-entropy between predicted and actual next tokens. Prompts masked with -100."},
    {"q": "What is quantization?", "a": "Reducing precision to save memory. 7B model: 14GB FP16 -> 3.5GB in 4-bit."},
    {"q": "What is temperature?", "a": "Controls randomness: p=exp(logit/T)/sum. High T=random, Low T=deterministic."},
]


def make_dataset():
    os.makedirs("data", exist_ok=True)
    examples = []
    for c in CONVERSATIONS:
        examples.append({"instruction": c["q"], "output": c["a"]})
    while len(examples) < 50:
        for c in CONVERSATIONS:
            if len(examples) >= 50:
                break
            examples.append({
                "instruction": f"Tell me about {c['q'].lower().replace('?','')}?",
                "output": c["a"],
            })
    out = examples[:50]
    with open("data/tiny_conversations.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Data created ({len(out)} examples)")


if __name__ == "__main__":
    make_dataset()
