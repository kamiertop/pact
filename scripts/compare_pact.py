"""Compare original PACT with the three fixed R1 sweep finalists."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from sweep_r1 import BASELINE, PARAMETERS, run_trial, test_winner, write_json


GROUPS = {
    "original": BASELINE,
    "r1_ratio3_min1024": (0.0, 0.1, 0.01, 3.0, 1024),
    "r1_ratio6_min64": (0.0, 0.1, 0.01, 6.0, 64),
    "r1_ratio1_min1024": (0.0, 0.1, 0.01, 1.0, 1024),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--run_root", type=Path, default=Path("runs/pact_comparison"))
    parser.add_argument("--gpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 42, 73])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="same patience for every group (0 disables early stopping)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_events_num", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--prediction_thresh", type=float, default=0.9)
    parser.add_argument("--dry_run", action="store_true", help="print the plan without training")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_events_num) < 1:
        parser.error("epochs, batch size and maximum event count must be positive")
    if args.early_stopping_patience < 0:
        parser.error("early stopping patience must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.prediction_thresh <= 1:
        parser.error("learning rate must be finite and positive and threshold must be in [0, 1]")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32)")

    print(f"{len(GROUPS) * len(args.seeds)} runs | seeds={args.seeds} | "
          f"epochs={args.epochs} | patience={args.early_stopping_patience} | "
          f"threshold={args.prediction_thresh}", flush=True)
    for name, params in GROUPS.items():
        print(f"{name}: {dict(zip(PARAMETERS, params))}", flush=True)
    if args.dry_run:
        return
    if any(not (args.data_dir / split).is_dir() for split in ("train", "val", "test")):
        parser.error("--data_dir must contain train, val and test directories")

    args.run_root = args.run_root.resolve()
    # Finish training before testing; the test scores never change the plan.
    trained = {name: [] for name in GROUPS}
    for seed in args.seeds:
        for name, params in GROUPS.items():
            trained[name].append(run_trial(args, name, params, args.epochs, seed))

    groups = {}
    for name, runs in trained.items():
        group_args = argparse.Namespace(**vars(args))
        group_args.run_root = args.run_root / name
        test_winner(group_args, {"group": name, "params": runs[0]["params"], "runs": runs})
        groups[name] = json.loads((group_args.run_root / "test_summary.json").read_text(encoding="utf-8"))

    baseline = {row["seed"]: row["test_metrics"]["iou"] for row in groups["original"]["runs"]}
    for group in groups.values():
        deltas = {row["seed"]: row["test_metrics"]["iou"] - baseline[row["seed"]]
                  for row in group["runs"]}
        group["paired_iou_delta_vs_original"] = {
            "by_seed": deltas, "mean": mean(deltas.values()),
            "std": stdev(deltas.values()) if len(deltas) > 1 else None,
        }

    path = args.run_root / "comparison.json"
    protocol = {**vars(args), "data_dir": str(args.data_dir.resolve()), "run_root": str(args.run_root)}
    write_json(path, {"protocol": protocol, "groups": groups})
    print("Group | IoU | seg_acc | Pd | Fa | mean paired IoU delta (pp)")
    for name, group in groups.items():
        cells = []
        for metric, value in group["mean_test_metrics"].items():
            std = group["std_test_metrics"][metric]
            factor = 1 if metric == "fa" else 100
            cell = f"{value * factor:.5g}"
            if std is not None:
                cell += f" ± {std * factor:.5g}"
            cells.append(cell)
        delta = group["paired_iou_delta_vs_original"]["mean"] * 100
        print(" | ".join([name, *cells, f"{delta:+.3f}"]))
    print(f"Comparison: {path}")


if __name__ == "__main__":
    main()
