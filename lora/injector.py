import torch
import torch.nn as nn
from .layers import LoRALayer


def _is_target_linear(child, target_modules, full_name):
    if isinstance(child, nn.Linear):
        return any(full_name.endswith(t) for t in target_modules)
    if hasattr(child, "weight") and child.weight.dim() == 2:
        cls = type(child).__name__
        if cls == "Conv1D":
            return any(full_name.endswith(t) for t in target_modules)
    return False


def inject_lora(model: nn.Module, config: dict) -> nn.Module:
    lora_cfg = config["lora"]
    target_modules = lora_cfg["target"]
    r = lora_cfg["r"]
    alpha = lora_cfg["alpha"]
    dropout = lora_cfg["dropout"]

    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if _is_target_linear(child, target_modules, full_name):
                targets.append((module, child_name, child))

    for parent, child_name, original in targets:
        setattr(parent, child_name, LoRALayer(original, r=r, alpha=alpha, dropout=dropout))

    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        is_lora = "A" in name or "B" in name
        param.requires_grad_(is_lora)
        total += param.numel()
        if is_lora:
            trainable += param.numel()

    print(f"  Params: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")
    return model
