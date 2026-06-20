import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lora import inject_lora
from lora.layers import LoRALayer


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.examples = []
        with open("data/tiny_conversations.json") as f:
            data = json.load(f)
        for item in data:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
            full = prompt + item["output"]
            ids = tokenizer(full, max_length=max_length, truncation=True, return_tensors=None)["input_ids"]
            prompt_len = len(tokenizer(prompt, max_length=max_length, truncation=True, return_tensors=None)["input_ids"])
            self.examples.append({
                "input_ids": ids[:max_length],
                "labels": [-100] * prompt_len + ids[prompt_len:],
                "attention_mask": [1] * len(ids[:max_length]),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    return {
        "input_ids": nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
        "labels": nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        "attention_mask": nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
    }


@torch.no_grad()
def run_eval(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += model(**batch).loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def _save_model_unwrap_lora(model, tokenizer, path):
    from lora.layers import LoRALayer
    from transformers import AutoModelForCausalLM

    config = load_config()
    fresh = AutoModelForCausalLM.from_pretrained(config["model"]["name"])

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            merged = module.merge_weights()
            parent_fresh = fresh
            parts = name.split(".")
            for p in parts[:-1]:
                parent_fresh = getattr(parent_fresh, p)
            child_name = parts[-1]
            original = getattr(parent_fresh, child_name)
            original.weight = torch.nn.Parameter(merged)

    fresh.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train():
    config = load_config()
    tc = config["training"]
    vc = config["validation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
    model = inject_lora(model, config)
    model = model.to(device)

    if not os.path.exists("data/tiny_conversations.json"):
        from make_dataset import make_dataset
        make_dataset()

    dataset = ConversationDataset(tokenizer, tc["max_length"])
    val_size = int(len(dataset) * vc["split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, collate_fn=collate_fn) if val_size else None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * tc["num_epochs"])

    use_amp = tc["mixed_precision"] != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if tc["mixed_precision"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(tc["mixed_precision"] == "fp16" and device.type == "cuda"))

    print(f"Training {tc['num_epochs']} epochs...\n")
    best_loss = float("inf")

    for epoch in range(tc["num_epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch).loss

            if tc["mixed_precision"] == "fp16" and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % tc["gradient_accumulation_steps"] == 0:
                if tc["gradient_clip"] > 0:
                    if tc["mixed_precision"] == "fp16" and device.type == "cuda":
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])

                if tc["mixed_precision"] == "fp16" and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})

        avg_loss = epoch_loss / len(train_loader)

        if val_loader and (epoch + 1) % max(1, int(1.0 / vc["eval_every_epochs"])) == 0:
            val_loss = run_eval(model, val_loader, device)
            print(f"  Val loss: {val_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("output/best_model", exist_ok=True)
            _save_model_unwrap_lora(model, tokenizer, "output/best_model")
            print(f"  Saved (loss: {avg_loss:.4f})")

    print(f"\nDone! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
