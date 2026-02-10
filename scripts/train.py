import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.config import load_config
from src.data.collate import collate_fn
from src.data.dataset import ImageTextDataset
from src.data.tokenizer import BasicTokenizer, build_vocab, load_vocab, save_vocab
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.clip import CLIPModel
from src.tracking.local_logger import LocalLogger
from src.tracking.mlflow_logger import MLflowLogger
from src.utils.seed import seed_everything
from src.train.trainer import train
from src.utils.dist import init_distributed, cleanup_distributed, is_distributed, is_main_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    init_distributed()

    data_root = cfg["paths"]["data_root"]
    train_file = cfg["paths"]["train_file"]
    val_file = cfg["paths"].get("val_file")
    output_dir = Path(cfg["paths"]["output_dir"])

    vocab_path = output_dir / "vocab.json"
    if is_main_process() and not vocab_path.exists():
        if Path(train_file).suffix.lower() == ".jsonl":
            import pandas as pd

            df = pd.read_json(train_file, lines=True)
        else:
            import pandas as pd

            df = pd.read_csv(train_file)
        vocab = build_vocab(df["text"].tolist(), cfg["model"]["text_vocab_size"])
        save_vocab(vocab, vocab_path)

    if is_distributed():
        dist.barrier()

    vocab = load_vocab(vocab_path)
    tokenizer = BasicTokenizer(vocab)

    train_tf = build_train_transform(
        cfg["augment"]["image_size"],
        cfg["augment"]["crop_scale_min"],
        cfg["augment"]["crop_scale_max"],
    )
    eval_tf = build_eval_transform(cfg["augment"]["image_size"])

    train_ds = ImageTextDataset(
        file_path=train_file,
        data_root=data_root,
        tokenizer=tokenizer,
        max_len=cfg["model"]["text_max_len"],
        transform=train_tf,
    )
    val_ds = None
    if val_file:
        val_ds = ImageTextDataset(
            file_path=val_file,
            data_root=data_root,
            tokenizer=tokenizer,
            max_len=cfg["model"]["text_max_len"],
            transform=eval_tf,
        )

    collate = lambda batch: collate_fn(batch, tokenizer.pad_id)

    train_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=train_sampler is None,
        pin_memory=True,
        collate_fn=collate,
        sampler=train_sampler,
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            shuffle=False,
            pin_memory=True,
            collate_fn=collate,
            sampler=val_sampler,
        )

    model = CLIPModel(
        image_backbone=cfg["model"]["image_encoder"],
        image_embed_dim=cfg["model"]["image_embed_dim"],
        text_vocab_size=cfg["model"]["text_vocab_size"],
        text_max_len=cfg["model"]["text_max_len"],
        text_embed_dim=cfg["model"]["text_embed_dim"],
        text_num_layers=cfg["model"]["text_num_layers"],
        text_num_heads=cfg["model"]["text_num_heads"],
        text_ffn_dim=cfg["model"]["text_ffn_dim"],
        text_dropout=cfg["model"]["text_dropout"],
        projection_dim=cfg["model"]["projection_dim"],
        temperature=cfg["train"]["temperature"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if is_distributed():
        model = DDP(model, device_ids=[int(torch.cuda.current_device())])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    local_logger = None
    if is_main_process() and (cfg["tracking"]["csv"] or cfg["tracking"]["jsonl"]):
        local_logger = LocalLogger(output_dir, cfg["tracking"]["csv"], cfg["tracking"]["jsonl"])

    mlflow_logger = None
    if is_main_process() and cfg["tracking"].get("mlflow", True):
        mlflow_logger = MLflowLogger(
            experiment_name=cfg["tracking"].get("experiment", "mm-embed"),
            run_name=cfg["tracking"].get("run_name"),
            tracking_uri=cfg["tracking"].get("tracking_uri"),
        )
        mlflow_logger.log_params({
            "batch_size": cfg["train"]["batch_size"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "projection_dim": cfg["model"]["projection_dim"],
            "image_encoder": cfg["model"]["image_encoder"],
            "text_layers": cfg["model"]["text_num_layers"],
            "accumulation_steps": cfg["train"]["accumulation_steps"],
        })

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        max_epochs=cfg["train"]["max_epochs"],
        log_every=cfg["train"]["log_every"],
        eval_every=cfg["train"]["eval_every"],
        use_amp=cfg["train"]["amp"],
        output_dir=output_dir,
        local_logger=local_logger,
        mlflow_logger=mlflow_logger,
        track_system_stats=cfg["tracking"]["system_stats"],
        grad_clip=cfg["train"]["grad_clip"],
        accumulation_steps=cfg["train"]["accumulation_steps"],
    )

    if local_logger:
        local_logger.close()
    if mlflow_logger:
        mlflow_logger.end()

    cleanup_distributed()


if __name__ == "__main__":
    main()
