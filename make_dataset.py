import json
import os

CONVERSATIONS = [
    # LoRA basics
    {"q": "What is LoRA?", "a": "LoRA (Low-Rank Adaptation) is a fine-tuning technique that freezes the original model weights and injects small trainable matrices A and B into attention layers. Instead of updating millions of parameters, you only train a tiny fraction — typically less than 1% — making fine-tuning fast and memory-efficient."},
    {"q": "How does LoRA work?", "a": "LoRA works by decomposing weight updates into two low-rank matrices. For a weight matrix W of size d×d, LoRA adds A (d×r) and B (r×d) where r is much smaller than d. The update becomes A×B instead of a full d×d matrix. This means you train far fewer parameters while still adapting the model."},
    {"q": "Why use LoRA instead of full fine-tuning?", "a": "LoRA uses significantly less memory since you only train a small fraction of parameters. It's faster, requires less GPU VRAM, and produces smaller adapter files. The original model weights stay frozen, so you can swap different LoRA adapters on the same base model."},
    {"q": "What is the rank in LoRA?", "a": "The rank (r) in LoRA determines the size of the low-rank matrices. A rank of 8 means the A and B matrices are much smaller than the original weight matrix. Higher rank means more capacity to learn but more parameters to train. Common ranks are 4, 8, 16, or 64."},
    {"q": "What is the alpha parameter in LoRA?", "a": "The alpha parameter in LoRA controls the scaling of the low-rank update. The actual update is scaled by alpha/rank. A higher alpha means the LoRA adaptation has more influence on the model's behavior. A common choice is alpha = 2 × rank."},
    {"q": "What are target modules in LoRA?", "a": "Target modules are the specific layers in a transformer where LoRA matrices are injected. Common targets include query (Q), key (K), value (V), and output projection layers. Targeting more layers gives more capacity but increases the number of trainable parameters."},

    # Fine-tuning
    {"q": "What is fine-tuning?", "a": "Fine-tuning is the process of continuing training a pretrained model on a specific dataset. The model learns domain-specific patterns from your data while retaining its general knowledge. It's like teaching a smart student about a new subject."},
    {"q": "What is the difference between fine-tuning and training from scratch?", "a": "Fine-tuning starts with a model that already understands language, grammar, and world knowledge. Training from scratch requires massive datasets and compute. Fine-tuning needs much less data and compute because the model already has foundational knowledge."},
    {"q": "What is catastrophic forgetting?", "a": "Catastrophic forgetting happens when a model loses its pretrained knowledge during fine-tuning. It learns the new data so well that it forgets what it knew before. LoRA helps prevent this by keeping the original weights frozen and only training the small adapter matrices."},
    {"q": "What is the learning rate in training?", "a": "The learning rate controls how much the model's weights change with each training step. Too high and the model overshoots optimal values and becomes unstable. Too low and training is very slow or gets stuck. Common values range from 1e-5 to 3e-4 for fine-tuning."},
    {"q": "What is gradient accumulation?", "a": "Gradient accumulation lets you simulate larger batch sizes by accumulating gradients over multiple forward passes before updating weights. If your GPU can only fit batch size 4 but you want effective batch size 16, you accumulate gradients for 4 steps before updating."},

    # Transformers and attention
    {"q": "What is a transformer?", "a": "A transformer is a neural network architecture based on self-attention mechanisms. It processes all input tokens in parallel rather than sequentially like RNNs. Transformers are the foundation of modern language models like GPT, BERT, and LLaMA."},
    {"q": "How does self-attention work?", "a": "Self-attention computes attention scores between every pair of tokens in a sequence. Each token generates a query (what am I looking for?), key (what do I contain?), and value (what information do I provide?). The attention weight is softmax(Q·K^T / sqrt(d)) · V, letting each token focus on relevant other tokens."},
    {"q": "What are Q, K, and V in attention?", "a": "Q (Query), K (Key), and V (Value) are three different projections of the input. The query asks 'what should I attend to?', the key says 'here is what I contain', and the value is the actual information passed along. Attention scores are computed between queries and keys, then used to weight the values."},
    {"q": "What is multi-head attention?", "a": "Multi-head attention runs multiple attention mechanisms in parallel, each with different learned projections. One head might learn syntactic relationships while another learns semantic ones. The outputs are concatenated and projected to produce the final attention output."},
    {"q": "What is causal masking?", "a": "Causal masking ensures each token can only attend to previous tokens, not future ones. This is essential for autoregressive language models like GPT, which generate text one token at a time. Without causal masking, the model could 'cheat' by seeing the answer during training."},

    # Model architecture
    {"q": "What is GPT-2?", "a": "GPT-2 is an autoregressive language model by OpenAI with 1.5 billion parameters (or 124M for the small version). It generates text by predicting the next token given previous tokens. GPT-2 was trained on diverse internet text and can perform various language tasks."},
    {"q": "What is a language model?", "a": "A language model predicts the probability of a sequence of words or tokens. It learns patterns in text — grammar, facts, reasoning — and can generate coherent text by sampling from these learned distributions. Models like GPT use transformer architectures for this."},
    {"q": "What is tokenization?", "a": "Tokenization converts text into numerical tokens that models can process. GPT-2 uses Byte-Pair Encoding (BPE), which breaks text into subword units. Common words become single tokens while rare words are split into parts. This balances vocabulary size with expressiveness."},
    {"q": "What is an embedding?", "a": "An embedding is a dense vector representation of a token. Similar tokens have similar embeddings. In GPT-2, the token embedding matrix maps token IDs to 768-dimensional vectors. These embeddings capture semantic meaning and are learned during training."},
    {"q": "What is a feed-forward network?", "a": "Feed-forward networks in transformers are simple two-layer neural networks applied independently to each position. They typically expand the dimension (e.g., 768 → 3072) then compress it back. They provide non-linear transformation capacity after attention."},

    # Training concepts
    {"q": "What is loss in training?", "a": "Loss measures how wrong the model's predictions are compared to the actual targets. For language models, it's typically cross-entropy loss — the difference between predicted token probabilities and the actual next token. Lower loss means better predictions."},
    {"q": "What is an epoch?", "a": "An epoch is one complete pass through the entire training dataset. If you have 1000 examples and batch size 100, one epoch is 10 training steps. Models typically train for multiple epochs, but too many can cause overfitting."},
    {"q": "What is overfitting?", "a": "Overfitting happens when a model memorizes the training data instead of learning general patterns. It performs well on training data but poorly on new data. Signs include decreasing training loss but increasing validation loss."},
    {"q": "What is a batch size?", "a": "Batch size is the number of training examples processed together in one forward pass. Larger batches give more stable gradient estimates but use more memory. Smaller batches are noisier but can generalize better. Common sizes are 8, 16, 32, or 64."},
    {"q": "What is gradient clipping?", "a": "Gradient clipping limits the maximum magnitude of gradients during backpropagation. If gradients become too large (exploding gradients), training becomes unstable. Clipping ensures gradients stay within a reasonable range, preventing wild weight updates."},

    # Mixed precision and optimization
    {"q": "What is mixed precision training?", "a": "Mixed precision training uses lower precision (FP16 or BF16) for some computations while keeping FP32 for weight updates. This reduces memory usage and speeds up training on modern GPUs with Tensor Cores. It typically gives 2-3x speedup with minimal accuracy loss."},
    {"q": "What is FP16 vs BF16?", "a": "FP16 (half precision) uses 16 bits for floating point numbers, giving range up to 65504. BF16 (brain float) also uses 16 bits but with a larger exponent range, making it more numerically stable. BF16 is preferred for training because it handles larger values without overflow."},
    {"q": "What is a cosine learning rate schedule?", "a": "A cosine schedule gradually reduces the learning rate following a cosine curve. It starts high for fast initial learning, then smoothly decreases. This is gentler than step-based schedules and often produces better final results. It's commonly used with Adam optimizer."},
    {"q": "What is the Adam optimizer?", "a": "Adam is an adaptive learning rate optimizer that maintains separate learning rates for each parameter. It combines momentum (like SGD with momentum) with RMSprop-style adaptive rates. It's the most popular optimizer for training neural networks due to its good default performance."},
    {"q": "What is weight decay?", "a": "Weight decay regularization adds a penalty proportional to the weight values during training. It encourages smaller weights, which helps prevent overfitting. In AdamW (Adam with decoupled weight decay), the decay is applied directly to weights rather than through gradients."},

    # Inference and generation
    {"q": "What is temperature in text generation?", "a": "Temperature controls randomness in text generation. It scales the logits before softmax. Temperature 1.0 uses the model's natural distribution. Higher temperature (e.g., 1.5) makes output more random and creative. Lower temperature (e.g., 0.3) makes output more deterministic and focused."},
    {"q": "What is top-k sampling?", "a": "Top-k sampling only considers the k most probable next tokens when generating text. For example, with k=50, only the 50 most likely next tokens are considered, and one is sampled. This prevents very unlikely tokens from being selected while maintaining diversity."},
    {"q": "What is top-p (nucleus) sampling?", "a": "Top-p sampling selects from the smallest set of tokens whose cumulative probability exceeds p. For example, with p=0.9, it considers the top tokens that together account for 90% of the probability mass. This dynamically adjusts the number of candidates based on the model's confidence."},
    {"q": "What is greedy decoding?", "a": "Greedy decoding always selects the single most probable next token. It's deterministic and fast but often produces repetitive, boring text because it never explores less likely but potentially better continuations."},
    {"q": "What is beam search?", "a": "Beam search maintains multiple candidate sequences (beams) simultaneously. At each step, it expands all beams and keeps the top-k most probable ones. This finds higher-probability outputs than greedy decoding but is slower. It's commonly used in translation models."},

    # Data and preprocessing
    {"q": "What is a dataset in machine learning?", "a": "A dataset is a collection of examples used to train and evaluate a model. In fine-tuning, it typically consists of input-output pairs. For a chat model, each example might be an instruction and its desired response. Good datasets are diverse, accurate, and representative of the target task."},
    {"q": "What is data preprocessing?", "a": "Data preprocessing transforms raw data into a format the model can consume. For text models, this includes tokenization (text to numbers), truncation (handling long sequences), padding (making batches uniform), and label masking (hiding prompt tokens from loss computation)."},
    {"q": "What is padding in NLP?", "a": "Padding adds special tokens (usually zeros) to make all sequences in a batch the same length. Since neural networks process batches of fixed-size tensors, shorter sequences need padding. An attention mask tells the model which tokens are real and which are padding."},
    {"q": "What is truncation?", "a": "Truncation cuts sequences that exceed the model's maximum length. Since transformers have a fixed context window (e.g., 512 tokens for GPT-2), longer inputs must be shortened. Smart truncation keeps the most important parts, usually the end of the input."},

    # Practical considerations
    {"q": "What is a checkpoint in training?", "a": "A checkpoint is a saved snapshot of the model during training. It includes the model weights, optimizer state, and training progress. Checkpoints let you resume training if interrupted and select the best model based on validation performance."},
    {"q": "What is validation loss?", "a": "Validation loss measures the model's performance on held-out data not seen during training. It's computed periodically during training to monitor overfitting. If training loss decreases but validation loss increases, the model is overfitting."},
    {"q": "How much GPU memory do I need for fine-tuning?", "a": "For GPT-2 (124M params) with LoRA, you need about 2-4GB VRAM. Full fine-tuning needs 6-8GB. For larger models like LLaMA-7B with LoRA, you need 16-24GB. CPU training works but is much slower — expect hours instead of minutes."},
    {"q": "Can I fine-tune on CPU?", "a": "Yes, but it's much slower. GPT-2 with LoRA trains at about 3-4 seconds per batch on CPU versus under 1 second on GPU. For small datasets (50-100 examples, 5 epochs), CPU training takes a few minutes. For larger datasets, GPU is strongly recommended."},
    {"q": "What is the difference between GPT-2 sizes?", "a": "GPT-2 comes in four sizes: Small (124M params), Medium (355M), Large (774M), and XL (1.5B). Larger models have more knowledge and better performance but require more memory and compute. Small is good for experimentation, XL for production quality."},
]


def make_dataset():
    os.makedirs("data", exist_ok=True)
    examples = []
    prefixes = [
        "Explain", "Describe", "What is", "How does", "Tell me about",
        "Can you explain", "What do you know about", "Define",
    ]
    for c in CONVERSATIONS:
        examples.append({"instruction": c["q"], "output": c["a"]})
        q_lower = c["q"].lower().rstrip("?")
        for prefix in prefixes[:2]:
            examples.append({"instruction": f"{prefix} {q_lower}?", "output": c["a"]})
    out = examples[:200]
    with open("data/tiny_conversations.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Data created ({len(out)} examples)")
