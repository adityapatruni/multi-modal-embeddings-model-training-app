import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.collate import collate_fn
from src.data.dataset import ImageTextDataset
from src.data.tokenizer import BasicTokenizer, load_vocab
from src.data.transforms import build_eval_transform
from src.models.clip import CLIPModel
from src.utils.metrics import retrieval_recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg["paths"]["data_root"]
    val_file = cfg["paths"].get("val_file")
    if val_file is None:
        raise ValueError("val_file is required for eval")

    output_dir = Path(cfg["paths"]["output_dir"])
    vocab = load_vocab(output_dir / "vocab.json")
    tokenizer = BasicTokenizer(vocab)

    eval_tf = build_eval_transform(cfg["augment"]["image_size"])
    val_ds = ImageTextDataset(
        file_path=val_file,
        data_root=data_root,
        tokenizer=tokenizer,
        max_len=cfg["model"]["text_max_len"],
        transform=eval_tf,
    )

    collate = lambda batch: collate_fn(batch, tokenizer.pad_id)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=False,
        pin_memory=True,
        collate_fn=collate,
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_sim = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits_i, _ = model(images, input_ids, attention_mask)
            all_sim.append(logits_i.detach().cpu())

    sim = torch.cat(all_sim, dim=0)
    r1 = retrieval_recall_at_k(sim, 1)
    r5 = retrieval_recall_at_k(sim, 5)
    r10 = retrieval_recall_at_k(sim, 10)
    print({"recall@1": r1, "recall@5": r5, "recall@10": r10})


if __name__ == "__main__":
    main()
