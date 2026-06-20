# Tiny LoRA Tuner

A beginner-friendly tool for fine-tuning GPT-2 using LoRA (Low-Rank Adaptation). Automatically scales to your hardware — works on Mac CPU, gaming GPUs, or beefy training rigs. Includes a web UI, CLI training, and interactive chat.

## What This Project Does

Fine-tunes a GPT-2 language model on 50+ Q&A topics covering LoRA, transformers, attention, training concepts, and more. Uses LoRA to train only ~1% of the model's parameters, making it efficient on any hardware.

**Auto-scaling:** The app detects your GPU/CPU and adjusts batch size, epochs, and precision automatically. No manual tuning needed.

**What you'll get:** A chatbot that answers questions about LLM/ML concepts with coherent, knowledgeable responses.

---

## Quick Start

```bash
cd ~/tiny-lora-tuner
source .venv/bin/activate

# Generate training data
python make_dataset.py

# Train the model (~30 seconds on CPU)
python train.py

# Chat with your model
python chat.py

# Or launch the web UI
python gradio_app.py
# Opens http://localhost:7860
```

---

## Project Structure

```
tiny-lora-tuner/
├── lora/
│   ├── __init__.py         # Exports LoRALayer, inject_lora
│   ├── layers.py           # LoRA layer implementation
│   └── injector.py         # Injects LoRA into model's attention layers
├── config.yaml             # All hyperparameters (lr, batch size, epochs, etc.)
├── make_dataset.py         # Creates 50 synthetic Q&A training examples
├── train.py                # CLI training script (run epochs, save best model)
├── chat.py                 # Terminal chatbot (loads trained model)
├── gradio_app.py           # Web UI with Config/Train/Chat tabs
├── requirements.txt        # Python dependencies
├── data/
│   └── tiny_conversations.json   # Generated training data
└── output/
    └── best_model/         # Saved fine-tuned model
```

---

## How It Works

### LoRA (Low-Rank Adaptation)

Instead of updating all 125M parameters in GPT-2, LoRA adds small trainable matrices (A and B) to the attention layers:

- **Original weights** → frozen (don't change)
- **LoRA matrices** → trainable (only 811K params)
- **Result:** Fast training on CPU with minimal memory

### Training Pipeline

1. `make_dataset.py` generates 50 Q&A pairs about LLM concepts
2. `train.py` loads GPT-2, injects LoRA layers, trains for 5 epochs
3. Best model (lowest loss) is saved to `output/best_model/`
4. Before saving, LoRA weights are merged back into the original model architecture

---

## Detailed Setup

### Prerequisites

- Python 3.10+
- macOS, Linux, or Windows (CPU-only, no GPU needed)
- ~500MB disk space (for model weights)

### Installation

```bash
# Navigate to project
cd ~/tiny-lora-tuner

# Create virtual environment
python3 -m venv .venv

# Activate it (MUST do this before every session)
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install torch transformers datasets pyyaml tqdm gradio matplotlib
```

### Why a virtual environment?

The `.venv` folder isolates this project's packages from your system Python. If you skip this step, you'll see `ModuleNotFoundError` errors.

---

## Usage Guide

### 1. Generate Training Data

```bash
source .venv/bin/activate
python make_dataset.py
```

Creates `data/tiny_conversations.json` with 50 Q&A pairs. You can edit `make_dataset.py` to add your own Q&A pairs.

### 2. CLI Training

```bash
python train.py
```

**What you'll see:**
```
Device: cpu
  Params: 125,250,816 total, 811,008 trainable (0.65%)
Training 5 epochs...

Epoch 1: 100%|██████████| 12/12 [00:03<00:00, 3.19it/s]
  Saved (loss: 5.4860)
...
Done! Best loss: 4.3930
```

**Key output:**
- `trainable (~1%)` — confirms LoRA is working (only tiny fraction trained)
- `GPU: XX GB` or `CPU mode` — auto-scaling message
- `Saved` — model saved to `output/best_model/`
- Loss should decrease across epochs (target: <3.0 for decent quality)

### 3. CLI Chat

```bash
python chat.py
```

```
Tiny LoRA Chat (type /exit to quit)

You: What is LoRA?
Bot: LoRA is a low-rank adaptation method that fine-tunes...

You: /exit
```

### 4. Web UI (Gradio)

```bash
python gradio_app.py
```

Opens browser at `http://localhost:7860` with 3 tabs:

| Tab | What it does |
|-----|-------------|
| **Config** | Set learning rate, batch size, epochs |
| **Train** | Click "Train" → watch loss decrease → see plot |
| **Chat** | Send messages, get responses from trained model |

**Note:** Keep the terminal running. Close it (Ctrl+C) to stop the server.

---

## Configuration

Edit `config.yaml` to change settings:

```yaml
model:
  name: gpt2              # Base model (gpt2 = 124M params)

lora:
  r: 16                   # Rank (higher = more capacity, more memory)
  alpha: 32               # Scaling factor (typically 2x rank)
  dropout: 0.05           # Regularization
  target:
    - c_attn              # GPT-2 attention layer names
    - c_proj

training:
  learning_rate: 0.0002   # How fast the model learns
  batch_size: 8           # Samples per training step
  num_epochs: 10          # How many times to loop through data
  max_length: 512         # Max token length per example
```

### Auto-Scaling

The app automatically detects your hardware and adjusts settings:

| Hardware | Batch Size | Epochs | Mixed Precision | Notes |
|----------|-----------|--------|-----------------|-------|
| **GPU 24GB+** (A100, 4090) | 32 | 15+ | BF16 | Full speed |
| **GPU 16GB** (3090, 4080) | 16 | 10 | BF16 | Great |
| **GPU 8GB** (3060, 4060) | 8 | 10 | FP16 | Good |
| **GPU <8GB** | 4 | 10 | FP16 | Reduced batch |
| **CPU** (Mac, Linux) | 4 | 10 | None | Slow but works |

Auto-scaling overrides `config.yaml` values at runtime. To force specific settings, set `batch_size` and `num_epochs` in the config.

**Common manual changes:**
- Training too slow? → Increase `batch_size` (needs more RAM)
- Loss not decreasing? → Try `learning_rate: 0.0001`
- Overfitting? → Increase `dropout: 0.1` or reduce `num_epochs`
- Better quality? → More training data + more epochs

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'yaml'` | Virtual env not activated | `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'transformers'` | Virtual env not activated | `source .venv/bin/activate` |
| `ValueError: optimizer got an empty parameter list` | Wrong target modules in config | Ensure `c_attn` and `c_proj` are in `config.yaml` |
| Training is very slow | Running on CPU (expected) | Normal for GPT-2 on CPU, ~3-4s per batch |
| `RuntimeError: size mismatch` when loading model | Model saved with wrong architecture | Re-run `python train.py` to retrain and save correctly |

---

## Customizing the Training Data

Edit `make_dataset.py` to add your own Q&A pairs:

```python
CONVERSATIONS = [
    {"q": "Your question here", "a": "Your answer here"},
    # Add more pairs...
]
```

The script generates 50 examples by rephrasing your base 10 pairs. After editing, re-run:

```bash
python make_dataset.py
python train.py
```

---

## Files You Might Edit

| File | When to edit |
|------|-------------|
| `config.yaml` | Change training hyperparameters |
| `make_dataset.py` | Add custom Q&A training data |
| `lora/layers.py` | Modify LoRA implementation (advanced) |
| `train.py` | Change training loop behavior (advanced) |

## Files You Should NOT Edit

| File | Why |
|------|-----|
| `lora/injector.py` | Handles GPT-2's Conv1D layers correctly — editing may break injection |
| `output/best_model/` | Auto-generated — gets overwritten on each training run |

---

## Key Concepts Glossary

| Term | Simple Explanation |
|------|-------------------|
| **LoRA** | Low-Rank Adaptation — trains tiny matrices instead of full model |
| **Fine-tuning** | Continuing training on your specific data |
| **GPT-2** | A 125M parameter language model by OpenAI |
| **Epoch** | One complete pass through the training data |
| **Loss** | How wrong the model is (lower = better) |
| **Learning Rate** | How big the steps are when updating weights |
| **Mixed Precision** | Using FP16/BF16 math for speed (auto-disabled on CPU) |
| **Conv1D** | GPT-2's linear layer format (transposed compared to nn.Linear) |
| **Merged weights** | LoRA weights folded back into the original model for clean saving |

---

## What Happens When You Train

```
1. Load GPT-2 (125M params, all frozen)
2. Inject LoRA layers into attention (c_attn, c_proj)
   → 811K new trainable params (0.65% of total)
3. For each epoch:
   a. Feed training Q&A pairs through model
   b. Compute loss (how wrong the predictions are)
   c. Backpropagate through LoRA layers only
   d. Update LoRA A and B matrices
   e. Save model if loss improved
4. Merge LoRA weights back into original GPT-2
5. Save clean model to output/best_model/
```

---

## Deploy to Cloud

The app is Docker-ready. Train and serve from any free cloud tier.

### Option 1: Render (Easiest — Free Tier)

1. Push to GitHub:
```bash
git add .
git commit -m "add deployment files"
git push
```

2. Go to [render.com](https://render.com), sign up, create a **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Runtime:** Docker
   - **Instance Type:** Free
   - **Port:** 7860
5. Deploy — takes ~2 minutes

### Option 2: Fly.io (Free Tier)

```bash
# Install flyctl
brew install flyctl

# Login
fly auth login

# Launch (from project root)
fly launch

# Deploy
fly deploy
```

Free tier: 3 shared-cpu-1x VMs, 256MB RAM each.

### Option 3: Railway (Free Tier)

```bash
# Install railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up
```

Free tier: $5 credit/month (enough for this app).

### Option 4: GCP Cloud Run (Free Tier)

```bash
# Install gcloud CLI
brew install google-cloud-sdk

# Auth
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tiny-lora-tuner
gcloud run deploy tiny-lora-tuner --image gcr.io/YOUR_PROJECT_ID/tiny-lora-tuner --port 7860 --allow-unauthenticated
```

Free tier: 180,000 vCPU-seconds, 360,000 GB-seconds per month.

### Option 5: Docker locally

```bash
docker-compose up --build
# Opens http://localhost:7860
```

### Model Size Warning

The model is **478MB**. Most free tiers have storage/bandwidth limits:
- **Render:** Free tier has 512MB disk — just fits
- **Fly.io:** Free tier has 3GB volume — plenty
- **Railway:** Depends on plan
- **Cloud Run:** No persistent storage by default

For production, consider pre-training locally and copying `output/best_model/` into the Docker image.

---

## License

Personal/educational use. GPT-2 weights are subject to OpenAI's license.
