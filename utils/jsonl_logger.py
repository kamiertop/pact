"""Append-only local training metrics, readable with the standard library."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


class JsonlLogger:
    def __init__(self, output_dir, config, seed):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        run_dir = Path(output_dir) / "logs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.path = run_dir / "metrics.jsonl"
        self.log("config", config=config, seed=seed)

    def log(self, event, **values):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        # Closing after every record keeps logs available if training is interrupted.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
