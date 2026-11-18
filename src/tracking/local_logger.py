from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


class LocalLogger:
    def __init__(self, output_dir: str | Path, enable_csv: bool, enable_jsonl: bool):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_csv = enable_csv
        self.enable_jsonl = enable_jsonl
        self.csv_path = self.output_dir / "metrics.csv"
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self._csv_file = None
        self._csv_writer = None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.enable_csv:
            if self._csv_writer is None:
                self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["step", *metrics.keys()])
                self._csv_writer.writeheader()
            row = {"step": step, **metrics}
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        if self.enable_jsonl:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
