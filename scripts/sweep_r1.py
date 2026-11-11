"""Screen R1 loss settings on validation IoU, then repeat finalists across seeds."""

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ("risk_weight", "iou_weight", "hardneg_weight", "hardneg_ratio", "hardneg_min")
BASELINE = (0.0, 0.0, 0.0, 3.0, 256)
EV_UAV = (0.1, 0.1, 0.03, 3.0, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--run_root", type=Path, default=Path("runs/r1_sweep"))
    parser.add_argument("--gpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_events_num", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--prediction_thresh", type=float, default=0.9)
    parser.add_argument("--screen_epochs", type=int, default=15)
    parser.add_argument("--final_epochs", type=int, default=50)
    parser.add_argument("--screen_seed", type=int, default=37)
    parser.add_argument("--final_seeds", nargs="+", type=int, default=[37, 42, 73])
    parser.add_argument("--risk_weights", nargs="+", type=float, default=[0, 0.05, 0.1])
    parser.add_argument("--iou_weights", nargs="+", type=float, default=[0, 0.05, 0.1])
    parser.add_argument("--hardneg_weights", nargs="+", type=float, default=[0, 0.01, 0.03])
    parser.add_argument("--hardneg_ratios", nargs="+", type=float, default=[1, 3, 6])
    parser.add_argument("--hardneg_mins", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--mining_top", type=int, default=2)
    parser.add_argument("--final_top", type=int, default=2,
                        help="additional finalists beyond baseline and EV-UAV settings")
    parser.add_argument("--stage", choices=("weights", "mining", "all"), default="all",
                        help="last stage to run")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if min(args.screen_epochs, args.final_epochs, args.batch_size, args.max_events_num) < 1:
        parser.error("epochs, batch size and maximum event count must be positive")
    if args.mining_top < 0 or args.final_top < 0:
        parser.error("candidate counts must be nonnegative")
    weights = args.risk_weights + args.iou_weights + args.hardneg_weights
    if any(not math.isfinite(w) or w < 0 for w in weights):
        parser.error("weights must be finite and nonnegative")
    if any(not math.isfinite(r) or r <= 0 for r in args.hardneg_ratios) or min(args.hardneg_mins) < 1:
        parser.error("mining ratios and minimum quotas must be positive and finite")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.prediction_thresh <= 1:
        parser.error("learning rate must be finite and positive and threshold must be in [0, 1]")
    if any(seed < 0 or seed >= 2**32 for seed in [args.screen_seed, *args.final_seeds]):
        parser.error("seeds must be in [0, 2**32)")
    if len(set(args.final_seeds)) != len(args.final_seeds):
        parser.error("final seeds must be distinct")
    return args


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_trial(args, phase, params, epochs, seed):
    config = dict(zip(PARAMETERS, params))
    options = {
        "data_dir": str(args.data_dir.resolve()), "batch_size": args.batch_size,
        "max_events_num": args.max_events_num, "lr": args.lr,
        "prediction_thresh": args.prediction_thresh, "epochs": epochs, "seed": seed,
        **config,
    }
    trial_id = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    directory = args.run_root.resolve() / phase / trial_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if Path(result["checkpoint"]).is_file():
            return result

    command = [sys.executable, "-u", str(ROOT / "train.py"),
               "--run_root", str(directory), "--run_name", phase]
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    if args.gpu is not None:
        command.extend(("--gpu", args.gpu))

    previous_runs = set(directory.glob("train_*"))
    print(f"{phase} | seed={seed} | {config} | {directory}", flush=True)
    log = directory / "train.log"
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise SystemExit(f"Training exited with code {process.returncode}. See {log}")

    run_dirs = set(directory.glob("train_*")) - previous_runs
    if len(run_dirs) != 1:
        raise SystemExit(f"Expected exactly one new training directory in {directory}")
    run_dir = run_dirs.pop()
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as stream:
        records = [record for line in stream if "val_mIoU" in (record := json.loads(line))]
    checkpoint = run_dir / "checkpoints" / f"best_iou_seed{seed}.pt"
    if (len(records) != epochs or not checkpoint.is_file()
            or any(not math.isfinite(row["val_mIoU"]) for row in records)):
        raise SystemExit(f"Missing checkpoint or incomplete validation metrics. See {log}")

    best = max(records, key=lambda row: row["val_mIoU"])
    result = {
        "params": config, "seed": seed, "epochs": epochs,
        "val_iou": best["val_mIoU"], "best_epoch": best["epoch"] + 1,
        "run_dir": str(run_dir), "checkpoint": str(checkpoint),
    }
    write_json(result_path, result)
    print(f"  best val IoU={result['val_iou']:.4f} at epoch {result['best_epoch']}", flush=True)
    return result


def main():
    args = parse_args()
    # Normalize numeric types so the same settings also reuse the same cached trial.
    weights = product(map(float, args.risk_weights), map(float, args.iou_weights),
                      map(float, args.hardneg_weights))
    grid = sorted({(*triple, *BASELINE[3:]) for triple in weights} | {BASELINE, EV_UAV})
    mining_grid = set(product(map(float, args.hardneg_ratios), args.hardneg_mins))
    max_mining = min(args.mining_top, sum(p[2] > 0 for p in grid)) * len(
        mining_grid - {BASELINE[3:]}
    )
    max_final = (2 + args.final_top) * len(args.final_seeds)
    if args.dry_run:
        for params in grid:
            print(dict(zip(PARAMETERS, params)))
        print(f"Weights: {len(grid)} runs x {args.screen_epochs} epochs")
        if args.stage != "weights":
            print(f"Mining: up to {max_mining} runs x {args.screen_epochs} epochs")
        if args.stage == "all":
            print(f"Confirmation: up to {max_final} runs x {args.final_epochs} epochs")
        return

    for split in ("train", "val"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir() or not any(split_dir.iterdir()):
            raise SystemExit(f"Missing or empty data split: {split_dir}")

    screened = {
        params: run_trial(args, "screen", params, args.screen_epochs, args.screen_seed)
        for params in grid
    }
    ranked = sorted(screened, key=lambda params: -screened[params]["val_iou"])
    write_json(args.run_root / "screening.json", [screened[p] for p in ranked])
    if args.stage == "weights":
        return

    mining_candidates = [p for p in ranked if p[2] > 0][:args.mining_top]
    for risk, iou, hardneg, _, _ in mining_candidates:
        for ratio, minimum in sorted(mining_grid):
            params = (risk, iou, hardneg, ratio, minimum)
            if params not in screened:
                screened[params] = run_trial(args, "screen", params, args.screen_epochs,
                                             args.screen_seed)

    ranked = sorted(screened, key=lambda params: -screened[params]["val_iou"])
    write_json(args.run_root / "screening.json", [screened[p] for p in ranked])
    if args.stage == "mining":
        return

    finalists = [BASELINE, EV_UAV] + [p for p in ranked if p not in {BASELINE, EV_UAV}][
        :args.final_top
    ]
    summary = []
    for params in finalists:
        results = [run_trial(args, "confirm", params, args.final_epochs, seed)
                   for seed in args.final_seeds]
        scores = [row["val_iou"] for row in results]
        summary.append({
            "params": results[0]["params"], "mean_val_iou": mean(scores),
            "std_val_iou": stdev(scores) if len(scores) > 1 else None, "runs": results,
        })
    summary.sort(key=lambda row: -row["mean_val_iou"])
    write_json(args.run_root / "confirmation.json", summary)
    print(f"Confirmation results: {args.run_root / 'confirmation.json'}", flush=True)


if __name__ == "__main__":
    main()
