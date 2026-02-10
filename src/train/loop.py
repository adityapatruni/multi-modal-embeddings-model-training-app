from __future__ import annotations

from typing import Dict

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.clip import clip_loss
from src.tracking.system_stats import get_system_stats
from src.utils.metrics import retrieval_recall_at_k
from src.utils.dist import is_main_process


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    log_every: int,
    use_amp: bool,
    local_logger,
    mlflow_logger,
    global_step: int,
    track_system_stats: bool,
    grad_clip: float,
    accumulation_steps: int,
):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", disable=not is_main_process())):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_i, logits_t = model(images, input_ids, attention_mask)
            loss = clip_loss(logits_i, logits_t) / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        if (step + 1) % log_every == 0 and is_main_process():
            avg = running_loss / log_every
            metrics = {"train/loss": avg}
            if track_system_stats:
                metrics.update(get_system_stats())
            if local_logger:
                local_logger.log_metrics(metrics, global_step)
            if mlflow_logger:
                mlflow_logger.log_metrics(metrics, global_step)
            running_loss = 0.0
        global_step += 1
    return global_step


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_image = []
    for batch in tqdm(loader, desc="eval", disable=not is_main_process()):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits_i, _ = model(images, input_ids, attention_mask)
        all_image.append(logits_i.detach().cpu())
    sim = torch.cat(all_image, dim=0)
    r1 = retrieval_recall_at_k(sim, 1)
    r5 = retrieval_recall_at_k(sim, 5)
    r10 = retrieval_recall_at_k(sim, 10)
    return {"val/recall@1": r1, "val/recall@5": r5, "val/recall@10": r10}
