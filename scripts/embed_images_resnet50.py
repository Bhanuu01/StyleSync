from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def _resolve_image_path(images_dir: Path, item_id: str) -> Path | None:
    file_id = str(item_id).zfill(10)
    subdir = file_id[:3]
    p = images_dir / subdir / f"{file_id}.jpg"
    if p.exists():
        return p
    return None


def _build_inputs(articles_csv: Path, images_dir: Path, max_items: int | None) -> Tuple[List[str], List[str]]:
    articles = pd.read_csv(articles_csv)
    if "article_id" not in articles.columns:
        raise ValueError("articles.csv must contain article_id")

    item_ids = articles["article_id"].astype(str).drop_duplicates().tolist()
    if max_items is not None:
        item_ids = item_ids[:max_items]

    kept_ids: List[str] = []
    kept_paths: List[str] = []
    for item_id in item_ids:
        p = _resolve_image_path(images_dir, item_id)
        if p is None:
            continue
        kept_ids.append(item_id)
        kept_paths.append(str(p))

    if not kept_ids:
        raise FileNotFoundError("No images found. Check --images-dir and dataset layout.")

    return kept_ids, kept_paths


def _load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--images-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="artifacts/image_embeddings_resnet50.npz")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-items", type=int, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir is not None else (data_dir / "images")

    item_ids, paths = _build_inputs(data_dir / "articles.csv", images_dir, args.max_items)

    ds = tf.data.Dataset.from_tensor_slices((paths, item_ids))
    ds = ds.map(lambda pth, iid: (_load_and_preprocess(pth), iid), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.applications.ResNet50(include_top=False, pooling="avg", weights="imagenet")

    out_ids: List[np.ndarray] = []
    out_embs: List[np.ndarray] = []

    steps = int(np.ceil(len(item_ids) / args.batch_size))
    for img_batch, id_batch in tqdm(ds, total=steps):
        emb = model(img_batch, training=False).numpy().astype(np.float32)
        ids = id_batch.numpy().astype(str)
        out_ids.append(ids)
        out_embs.append(emb)

    item_id_arr = np.concatenate(out_ids, axis=0)
    emb_arr = np.concatenate(out_embs, axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, item_id=item_id_arr, embedding=emb_arr)


if __name__ == "__main__":
    main()
