"""Run the PACT paper comparison with the common driver from the sibling EV-UAV repo."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    driver = repo.parent / "ev-uav/scripts/paper_ablation.py"
    if not driver.is_file():
        raise SystemExit(f"Place the updated ev-uav repository alongside pact. Missing {driver}")
    runpy.run_path(str(driver))["main"]("pact", repo)
