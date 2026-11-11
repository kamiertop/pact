"""Keep configuration, metrics, and checkpoints together for each run."""
from datetime import datetime
import json
from pathlib import Path


class RunManager:
    def __init__(self, root, name, config):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(root) / f"train_{name}_{stamp}"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True)
        self.log_path = self.run_dir / "metrics.jsonl"
        self.log_path.touch()
        (self.run_dir / "config.json").write_text(
            json.dumps(vars(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def log_metric(self, epoch, batch=None, **metrics):
        record = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        if batch is not None:
            record["batch"] = batch
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(self, model, filename):
        import torch
        torch.save(model.state_dict(), self.ckpt_dir / filename)


def resolve_checkpoint(root, run_dir=None):
    pattern = "checkpoints/best_iou_seed*.pt" if run_dir else "train_*/checkpoints/best_iou_seed*.pt"
    base = Path(run_dir or root)
    candidates = [p for p in base.glob(pattern) if p.is_file()]
    if not candidates:
        raise ValueError(f"No best-IoU checkpoint found in {base}. Train first or select an existing run with --run_dir.")
    return max(candidates, key=lambda p: (p.parent.parent.name[-22:], p.stat().st_mtime_ns, str(p)))
