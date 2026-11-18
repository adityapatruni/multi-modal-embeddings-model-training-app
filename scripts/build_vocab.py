import argparse
from pathlib import Path

import pandas as pd

from src.data.tokenizer import build_vocab, save_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV or JSONL")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--out", required=True, help="Output vocab json")
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix.lower() == ".jsonl":
        df = pd.read_json(data_path, lines=True)
    else:
        df = pd.read_csv(data_path)

    vocab = build_vocab(df["text"].tolist(), args.vocab_size)
    save_vocab(vocab, args.out)


if __name__ == "__main__":
    main()
