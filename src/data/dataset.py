from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .tokenizer import BasicTokenizer


@dataclass
class Sample:
    image_path: str
    text: str


class ImageTextDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        data_root: str | Path,
        tokenizer: BasicTokenizer,
        max_len: int,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Sample]:
        if self.file_path.suffix.lower() == ".jsonl":
            df = pd.read_json(self.file_path, lines=True)
        else:
            df = pd.read_csv(self.file_path)
        samples = []
        for _, row in df.iterrows():
            samples.append(Sample(image_path=row["image_path"], text=row["text"]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image_path = self.data_root / sample.image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        input_ids = self.tokenizer.encode(sample.text, self.max_len)
        return {
            "image": image,
            "input_ids": input_ids,
            "text": sample.text,
        }
