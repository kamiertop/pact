"""EV-UAV R1 auxiliary losses, averaged equally over event samples."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class R1Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weights = {
            "risk": cfg.risk_weight,
            "iou": cfg.iou_weight,
            "hardneg": cfg.hardneg_weight,
        }
        self.ratio = cfg.hardneg_ratio
        self.minimum = cfg.hardneg_min

    def forward(self, logits, labels, sample_ids, instance_ids):
        logits = logits.reshape(-1)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        foreground = labels.reshape(-1) > 0.5
        zero = logits[:0].sum()
        losses = {name: [] for name, weight in self.weights.items() if weight}
        for sample in torch.unique(sample_ids, sorted=True):
            mask = sample_ids == sample
            sample_logits = logits[mask]
            positives = foreground[mask]
            if "risk" in losses:
                eligible = positives & (instance_ids[mask] > 0)
                positive_logits = sample_logits[eligible]
                losses["risk"].append(
                    F.softplus(-positive_logits).mean() if positive_logits.numel() else zero
                )
            if "iou" in losses:
                if positives.any():
                    probs = sample_logits.sigmoid()
                    labels_float = positives.to(probs.dtype)
                    intersection = (probs * labels_float).sum()
                    union = (probs + labels_float - probs * labels_float).sum()
                    losses["iou"].append(1 - intersection / union.clamp_min(1e-7))
                else:
                    losses["iou"].append(F.softplus(sample_logits).mean())
            if "hardneg" in losses:
                background = sample_logits[~positives]
                quota = max(self.minimum, math.ceil(self.ratio * positives.sum().item()))
                k = min(background.numel(), quota)
                selected = background.topk(k, sorted=False).values
                losses["hardneg"].append(F.softplus(selected).mean() if k else zero)

        components = {
            name: torch.stack(values).mean() if values else zero
            for name, values in losses.items()
        }
        total = sum((self.weights[name] * value for name, value in components.items()), zero)
        return total, components
