from __future__ import annotations

import os
import argparse
from pathlib import Path

import tensorflow as tf

from fashion_recsys.utils.embedding_export import export_user_embedding
from fashion_recsys.utils.faiss_index import FaissArtifacts, load_ip_index, search


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default="artifacts/retrieval_model")
    p.add_argument("--faiss-dir", type=str, default="artifacts/faiss")
    p.add_argument("--user-id", type=str, required=True)
    p.add_argument("--age", type=float, default=None)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    user_tower = tf.keras.models.load_model(model_dir / "user_tower")

    feats = {"user_id": args.user_id}
    if args.age is not None:
        feats["age"] = float(args.age)

    user_emb = export_user_embedding(user_tower, feats)

    artifacts = FaissArtifacts(index_path=Path(args.faiss_dir) / "faiss.index", ids_path=Path(args.faiss_dir) / "item_ids.npy")
    index, item_ids = load_ip_index(artifacts)

    rec_ids, scores = search(index=index, item_ids=item_ids, query_embeddings=user_emb, top_k=args.top_k)

    for i in range(rec_ids.shape[1]):
        print(str(rec_ids[0, i]), float(scores[0, i]))


if __name__ == "__main__":
    main()
