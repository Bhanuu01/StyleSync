from __future__ import annotations

import os
import argparse
import shutil
from pathlib import Path

import tensorflow as tf

from fashion_recsys.data.hm import build_tf_datasets, get_vocabularies, load_hm_frames, load_image_embeddings_npz
from fashion_recsys.models.two_tower import ItemTower, TwoTowerRetrievalModel, UserTower


class _UserTowerServingModule(tf.Module):
    def __init__(self, user_tower: tf.keras.Model):
        super().__init__()
        self.user_tower = user_tower

    @tf.function(
        input_signature=[
            {
                "user_id": tf.TensorSpec(shape=[None], dtype=tf.string, name="user_id"),
                "age": tf.TensorSpec(shape=[None], dtype=tf.float32, name="age"),
            }
        ]
    )
    def __call__(self, features):
        emb = self.user_tower({"user_id": features["user_id"], "age": features["age"]})
        return {"embedding": emb}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="artifacts/retrieval_model")
    p.add_argument("--export-dir", type=str, default="artifacts/model")
    p.add_argument("--image-embeddings", type=str, default=None)
    p.add_argument("--transactions-nrows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    frames = load_hm_frames(args.data_dir, interactions_nrows=args.transactions_nrows)
    user_vocab, item_vocab = get_vocabularies(frames)

    image_embeddings = load_image_embeddings_npz(args.image_embeddings) if args.image_embeddings else None
    train_ds, test_ds, item_ds = build_tf_datasets(
        frames,
        batch_size=args.batch_size,
        image_embeddings=image_embeddings,
    )

    user_tower = UserTower(user_vocab=user_vocab, embedding_dim=args.embedding_dim)

    if frames.customers is not None and "age" in frames.customers.columns:
        age_ds = train_ds.map(lambda x: x["age"])  # type: ignore[no-redef]
        user_tower.adapt(age_ds)

    item_tower = ItemTower(item_vocab=item_vocab, embedding_dim=args.embedding_dim)
    text_ds = item_ds.map(lambda x: x["item_text"])
    item_tower.adapt(text_ds)

    model = TwoTowerRetrievalModel(user_tower=user_tower, item_tower=item_tower, candidate_dataset=item_ds)

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    if hasattr(tf.keras.optimizers, "legacy"):
        optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=0.1)
    model.compile(optimizer=optimizer)

    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)

    model_dir = Path(args.model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    sample = next(iter(train_ds.take(1)))
    _ = user_tower({k: sample[k] for k in sample if k in {"user_id", "age"}})
    _ = item_tower({k: sample[k] for k in sample if k in {"item_id", "item_text", "image_embedding"}})

    user_tower.save(model_dir / "user_tower", include_optimizer=False)
    item_tower.save(model_dir / "item_tower", include_optimizer=False)

    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    export_path = export_root / "user_tower"
    if export_path.exists():
        shutil.rmtree(export_path)
    serving_module = _UserTowerServingModule(user_tower=user_tower)
    tf.saved_model.save(serving_module, str(export_path), signatures={"serving_default": serving_module.__call__})


if __name__ == "__main__":
    main()
