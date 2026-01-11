from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fashion_recsys.utils.faiss_index import build_ip_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=str, default="artifacts/item_embeddings.npz")
    p.add_argument("--output-dir", type=str, default="artifacts/faiss")
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()

    data = np.load(args.embeddings, allow_pickle=False)
    item_ids = data["item_id"]
    embs = data["embedding"]

    build_ip_index(item_ids=item_ids, item_embeddings=embs, output_dir=Path(args.output_dir), normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
