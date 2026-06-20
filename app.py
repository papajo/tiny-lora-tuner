import os
import yaml
import torch
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from lora import inject_lora
from lora.layers import LoRALayer
from train import ConversationDataset, collate_fn, run_eval, _save_model_unwrap_lora, auto_scale_config, load_config


state = {"model": None, "tokenizer": None, "trained": False}


def train_fn(lr, bs, epochs):
    config = load_config()
    config = auto_scale_config(config)
    config["training"]["learning_rate"] = lr
    config["training"]["batch_size"] = int(bs)
    config["training"]["num_epochs"] = int(epochs)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
    model = inject_lora(model, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state["model"] = model
    state["tokenizer"] = tokenizer

    if not os.path.exists("data/tiny_conversations.json"):
        from make_dataset import make_dataset
        make_dataset()

    dataset = ConversationDataset(tokenizer, config["training"]["max_length"])
    val_size = int(len(dataset) * config["validation"]["split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=int(bs), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(bs), shuffle=False, collate_fn=collate_fn) if val_size else None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * int(epochs))

    losses = []
    best_loss = float("inf")

    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            if (step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                if config["training"]["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            losses.append((len(losses), loss.item()))

        avg = epoch_loss / len(train_loader)
        if avg < best_loss:
            best_loss = avg
            os.makedirs("output/best_model", exist_ok=True)
            _save_model_unwrap_lora(model, tokenizer, "output/best_model")

    state["trained"] = True
    fig, ax = plt.subplots()
    ax.plot(*zip(*losses), label="Train")
    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    return f"Done! Best loss: {best_loss:.4f}", fig


def chat_fn(msg, history):
    if not state["trained"] and state["model"] is None:
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": "Train the model first!"})
        return "", history

    if state["model"] is None:
        model = AutoModelForCausalLM.from_pretrained("output/best_model")
        tokenizer = AutoTokenizer.from_pretrained("output/best_model")
        state["model"] = model
        state["tokenizer"] = tokenizer

    prompt = f"### Instruction:\n{msg}\n\n### Response:\n"
    inputs = state["tokenizer"](prompt, return_tensors="pt")
    with torch.no_grad():
        output = state["model"].generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=state["tokenizer"].eos_token_id,
        )
    response = state["tokenizer"].decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": response})
    return "", history


with gr.Blocks(title="Tiny LoRA Tuner") as app:
    gr.Markdown("# Tiny LoRA Tuner")
    with gr.Tab("Config"):
        lr = gr.Number(3e-4, label="Learning Rate")
        bs = gr.Slider(1, 32, 4, step=1, label="Batch Size")
        ep = gr.Slider(1, 10, 3, step=1, label="Epochs")
    with gr.Tab("Train"):
        train_btn = gr.Button("Train")
        status = gr.Textbox()
        plot = gr.Plot()
        train_btn.click(fn=train_fn, inputs=[lr, bs, ep], outputs=[status, plot])
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        msg.submit(fn=chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
