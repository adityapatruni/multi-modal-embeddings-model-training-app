from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import mlflow


class MLflowLogger:
    def __init__(self, experiment_name: str, run_name: Optional[str], tracking_uri: Optional[str]):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, object]) -> None:
        flat = {}
        for k, v in params.items():
            flat[str(k)] = v
        mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def end(self) -> None:
        mlflow.end_run()
