from __future__ import annotations

import os
import argparse
from pathlib import Path

import tensorflow as tf

from fashion_recsys.data.hm import build_item_dataset, load_hm_articles, load_image_embeddings_npz
from fashion_recsys.utils.embedding_export import export_item_embeddings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="artifacts/retrieval_model")
    p.add_argument("--output", type=str, default="artifacts/item_embeddings.npz")
    p.add_argument("--image-embeddings", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()

    image_embeddings = load_image_embeddings_npz(args.image_embeddings) if args.image_embeddings else None
    articles = load_hm_articles(args.data_dir)
    item_ds = build_item_dataset(articles=articles, batch_size=args.batch_size, image_embeddings=image_embeddings)

    model_dir = Path(args.model_dir)
    item_tower = tf.keras.models.load_model(model_dir / "item_tower")

    out = Path(args.output)
    export_item_embeddings(item_tower=item_tower, item_dataset=item_ds, output_path=out)


if __name__ == "__main__":
    main()
