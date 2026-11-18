from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.train.loop import evaluate, train_one_epoch
from src.utils.dist import is_main_process


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: torch.device,
    max_epochs: int,
    log_every: int,
    eval_every: int,
    use_amp: bool,
    output_dir: Path,
    local_logger,
    mlflow_logger,
    track_system_stats: bool,
    grad_clip: float,
    accumulation_steps: int,
):
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler(enabled=use_amp)
    global_step = 0
    best_r1 = 0.0

    for epoch in range(1, max_epochs + 1):
        global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            log_every,
            use_amp,
            local_logger,
            mlflow_logger,
            global_step,
            track_system_stats,
            grad_clip,
            accumulation_steps,
        )

        if epoch % eval_every == 0 and val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            if is_main_process():
                if local_logger:
                    local_logger.log_metrics(metrics, global_step)
                if mlflow_logger:
                    mlflow_logger.log_metrics(metrics, global_step)
                r1 = metrics["val/recall@1"]
                if r1 > best_r1:
                    best_r1 = r1
                    ckpt = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_r1": best_r1,
                    }
                    ckpt_path = output_dir / "best.pt"
                    torch.save(ckpt, ckpt_path)
                    if mlflow_logger:
                        mlflow_logger.log_artifact(ckpt_path)

    if is_main_process():
        ckpt_path = output_dir / "last.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        if mlflow_logger:
            mlflow_logger.log_artifact(ckpt_path)
