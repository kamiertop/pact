"""Command-line options shared by PACT training and evaluation."""

import argparse
import json
import math
from pathlib import Path

from utils.run_manager import resolve_checkpoint


def parse(mode):
    parser = argparse.ArgumentParser(
        description=f"PACT {mode}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="dataset/EV-UAV-dataset",
                        help="dataset root containing train, val and test")
    parser.add_argument("--gpu", help="CUDA device index (defaults to the current CUDA visibility)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_channel", type=int, default=4)
    parser.add_argument("--whole_t", type=int, default=8000, help="time window in ms")
    parser.add_argument("--res", nargs=2, type=int, default=[352, 288],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--prediction_thresh", type=float, default=0.9)
    parser.add_argument("--run_root", default="runs", help="root directory for training runs")
    if mode == "train":
        parser.add_argument("--seed", type=int, default=37)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--k", type=int, default=3)
        parser.add_argument("--t", type=int, default=5)
        parser.add_argument("--max_events_num", type=int, default=100000)
        parser.add_argument("--risk_weight", type=float, default=0.0,
                            help="R1 foreground event BCE weight")
        parser.add_argument("--iou_weight", type=float, default=0.0,
                            help="event Soft Jaccard weight")
        parser.add_argument("--hardneg_weight", type=float, default=0.0,
                            help="hard background BCE weight")
        parser.add_argument("--hardneg_ratio", type=float, default=3.0,
                            help="hard background events per foreground event")
        parser.add_argument("--hardneg_min", type=int, default=256,
                            help="minimum hard background quota per sample")
        parser.add_argument("--run_name", default="experiment", help="label for this training run")
    else:
        parser.add_argument("--run_dir", help="use this run's best validation IoU checkpoint")
        parser.add_argument("--split", choices=("val", "test"), default="test")
        parser.add_argument("--output_json", type=Path, help="write evaluation metrics as JSON")
        parser.add_argument("--pd_detT", type=int, default=50, help="Pd/Fa window in ms")
        parser.add_argument("--correct_thresh", type=float, default=0.0001)
        parser.add_argument("--roc", action=argparse.BooleanOptionalAction, default=True,
                            help="compute Pd and Fa")
    cfg = parser.parse_args()
    if mode == "train" and (not cfg.run_name or Path(cfg.run_name).name != cfg.run_name
                            or cfg.run_name in {".", ".."}):
        parser.error("--run_name must be a single directory name")
    if mode == "train":
        for name in ("risk_weight", "iou_weight", "hardneg_weight"):
            weight = getattr(cfg, name)
            if not math.isfinite(weight) or weight < 0:
                parser.error(f"--{name} must be finite and nonnegative")
        if not math.isfinite(cfg.hardneg_ratio) or cfg.hardneg_ratio <= 0:
            parser.error("--hardneg_ratio must be finite and positive")
        if cfg.hardneg_min < 1:
            parser.error("--hardneg_min must be positive")
    if mode == "test":
        try:
            checkpoint = resolve_checkpoint(cfg.run_root, cfg.run_dir)
            config_path = checkpoint.parent.parent / "config.json"
            if checkpoint.parent.name == "checkpoints" and config_path.is_file():
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                shared = ("data_dir", "batch_size", "input_channel", "whole_t",
                          "res", "width", "prediction_thresh")
                parser.set_defaults(**{k: saved[k] for k in shared if k in saved})
                cfg = parser.parse_args()
            cfg.checkpoint_path = checkpoint
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if not 0 <= cfg.prediction_thresh <= 1:
        parser.error("--prediction_thresh must be between 0 and 1")
    return cfg
