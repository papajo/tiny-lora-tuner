# Tiny LoRA Tuner — Lab Guide

Hands-on exercises to learn LoRA fine-tuning by doing.

---

## Lab 1: Run It

**Goal:** Train a GPT-2 model with LoRA on your machine.

```bash
cd ~/tiny-lora-tuner
source .venv/bin/activate
python make_dataset.py
python train.py
python chat.py
```

**What to observe:**
- The `Params` line shows only a tiny fraction of parameters are trainable (~0.24–0.65%)
- Training completes in under a minute on CPU
- Loss decreases across epochs
- The chatbot answers questions about LoRA/ML concepts

**Try:** Run `python chat.py` and ask "What is LoRA?" — compare the answer quality to a base GPT-2 by commenting out `inject_lora` and retraining.

---

## Lab 2: Read the LoRA Math

**Goal:** Understand what happens inside `lora/layers.py`.

Read `lora/layers.py` and answer these questions:

1. What are matrices `A` and `B` initialized to, and why? (Hint: `torch.randn` × 0.02 vs `torch.zeros`)
2. What does `self.scale = alpha / r` do to the LoRA output?
3. Why is `self.W.requires_grad_(False)` important? What happens if you forget it?
4. What is the `merge_weights()` function doing mathematically?

**Bonus:** Draw the forward pass on paper:
```
x → W (frozen) → output_1
x → A → B → scale → dropout → output_2
result = output_1 + output_2
```

---

## Lab 3: Tweak the Params

**Goal:** See how hyperparameters affect training.

Edit `config.yaml` and retrain after each change:

| Experiment | Config change | What to watch |
|-----------|--------------|---------------|
| Rank matters | `r: 4` then `r: 32` | Parameter count, training speed, final loss |
| Learning rate | `learning_rate: 0.001` then `0.00001` | Loss curve shape (too high = unstable, too low = slow) |
| More data | Add 10 entries to `make_dataset.py` | Does more data improve chatbot quality? |
| Dropout | `dropout: 0.0` then `0.3` | Overfitting vs underfitting |
| Batch size | `batch_size: 2` then `16` | Training stability, speed |

**Record your results:** After each experiment, note the best loss and whether the chatbot answers improved.

---

## Lab 4: Compare to H2O LLM Studio

**Goal:** Compare this LoRA approach to a full fine-tuning tool.

1. Go to [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) — a GUI tool for fine-tuning LLMs
2. Explore the UI and note what options it offers that this project doesn't
3. Questions to consider:
   - What does H2O LLM Studio handle that `train.py` does manually?
   - What LoRA settings does it expose?
   - Could you use this project's `output/best_model/` in H2O LLM Studio for inference?

**Reflection:** Both tools fine-tune LLMs, but at very different scales. This project teaches the fundamentals; H2O LLM Studio is production-ready. Understanding the fundamentals helps you use the production tools better.

---

## What You Learned

| Lab | Skill |
|-----|-------|
| 1 | Running a training pipeline end-to-end |
| 2 | Reading and understanding LoRA implementation |
| 3 | Hyperparameter experimentation and debugging |
| 4 | Comparing tools and understanding tradeoffs |
