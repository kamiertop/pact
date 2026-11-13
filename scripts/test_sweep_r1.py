"""Test completed sweep runs whose best validation IoU exceeds a cutoff."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from sweep_r1 import ROOT, write_json

METRICS = ("iou", "seg_acc", "pd", "fa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=Path("runs/r1_sweep"))
    parser.add_argument("--data_dir", type=Path, help="override the saved dataset path")
    parser.add_argument("--gpu", help="CUDA device passed to test.py")
    parser.add_argument("--min_val_iou", type=float, default=0.7)
    parser.add_argument("--dry_run", action="store_true", help="list candidates without testing")
    args = parser.parse_args()
    if not 0 <= args.min_val_iou <= 1:
        parser.error("--min_val_iou must be between 0 and 1")

    root = args.run_root.resolve()
    screening = root / "screening.json"
    confirmation = root / "confirmation.json"
    if not screening.is_file():
        parser.error(f"missing sweep summary: {screening}")
    candidates = [("screen", run) for run in json.loads(screening.read_text())]
    if confirmation.is_file():
        candidates.extend(
            ("confirm", run)
            for group in json.loads(confirmation.read_text())
            for run in group["runs"]
        )

    selected = {}
    for phase, run in candidates:
        if run["val_iou"] > args.min_val_iou:
            original = Path(run["run_dir"])
            run_dir = root / phase / original.parent.name / original.name
            selected[run_dir] = {**run, "phase": phase, "run_dir": str(run_dir)}

    print(f"Selected {len(selected)} runs with best val IoU > {args.min_val_iou:g}", flush=True)
    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    ranking = []
    failures = []
    ordered = sorted(selected.items(), key=lambda item: -item[1]["val_iou"])
    for index, (run_dir, run) in enumerate(ordered, 1):
        print(
            f"[{index}/{len(ordered)}] {run['phase']} seed={run['seed']} "
            f"val IoU={run['val_iou']:.6f} epoch={run['best_epoch']} | {run_dir}",
            flush=True,
        )
        if args.dry_run:
            continue

        checkpoint = run_dir / "checkpoints" / f"best_iou_seed{run['seed']}.pt"
        log = run_dir / "sweep_candidate_test.log"
        output = run_dir / "sweep_candidate_test.json"
        try:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            config = json.loads((run_dir / "config.json").read_text())
            command = [
                sys.executable, "-u", str(ROOT / "test.py"),
                "--split", "test", "--batch_size", "1",
                "--run_dir", str(run_dir),
                "--prediction_thresh", str(config["prediction_thresh"]),
                "--output_json", str(output),
            ]
            if data_dir is not None:
                command.extend(("--data_dir", str(data_dir)))
            if args.gpu is not None:
                command.extend(("--gpu", args.gpu))
            output.unlink(missing_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"test.py exited with code {process.returncode}, see {log}")
            metrics = json.loads(output.read_text())
            if not all(math.isfinite(metrics[name]) for name in METRICS):
                raise ValueError(f"non-finite test metric in {output}")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            failures.append({**run, "checkpoint": str(checkpoint), "error": str(exc)})
            print(f"  Failed: {exc}", flush=True)
            continue

        ranking.append({
            **run,
            "checkpoint": str(checkpoint),
            "prediction_thresh": config["prediction_thresh"],
            "test_metrics": metrics,
        })
        print("  " + " | ".join(f"{name}={metrics[name]:.6g}" for name in METRICS), flush=True)

    if args.dry_run:
        return 0

    ranking.sort(key=lambda run: -run["test_metrics"]["iou"])
    summary = root / "test_ranking.json"
    write_json(summary, {
        "min_val_iou_exclusive": args.min_val_iou,
        "selected_count": len(selected),
        "ranking": ranking,
        "failures": failures,
    })
    if ranking:
        best = ranking[0]
        print(f"Best test IoU={best['test_metrics']['iou']:.6f} | {best['run_dir']}")
        print(f"Parameters: {best['params']}")
    print(f"Tested {len(ranking)}/{len(selected)} runs, {len(failures)} failed. Results: {summary}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
