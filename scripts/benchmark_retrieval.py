from __future__ import annotations

import os
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from fashion_recsys.utils.embedding_export import export_user_embedding
from fashion_recsys.utils.faiss_index import FaissArtifacts, load_ip_index, search


def _percentiles_ms(values_s: np.ndarray) -> dict:
    ms = values_s * 1000.0
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="artifacts/retrieval_model")
    p.add_argument("--faiss-dir", type=str, default="artifacts/faiss")
    p.add_argument("--transactions-nrows", type=int, default=200000)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    model_dir = Path(args.model_dir)
    user_tower = tf.keras.models.load_model(model_dir / "user_tower")

    artifacts = FaissArtifacts(index_path=Path(args.faiss_dir) / "faiss.index", ids_path=Path(args.faiss_dir) / "item_ids.npy")
    index, item_ids = load_ip_index(artifacts)

    tx_path = Path(args.data_dir) / "transactions_train.csv"
    tx = pd.read_csv(tx_path, nrows=args.transactions_nrows)
    if "customer_id" not in tx.columns:
        raise ValueError("transactions_train.csv must contain customer_id")

    users = tx["customer_id"].astype(str).drop_duplicates().to_numpy()
    if len(users) == 0:
        raise ValueError("No users found in transactions sample")

    user_ids = rng.choice(users, size=min(args.n_queries, len(users)), replace=len(users) < args.n_queries)

    warmup = min(10, len(user_ids))
    for i in range(warmup):
        _ = export_user_embedding(user_tower, {"user_id": str(user_ids[i])})

    embed_times = []
    faiss_times = []
    end_to_end_times = []

    for uid in user_ids:
        t0 = time.perf_counter()
        emb = export_user_embedding(user_tower, {"user_id": str(uid)})
        t1 = time.perf_counter()
        _rec_ids, _scores = search(index=index, item_ids=item_ids, query_embeddings=emb, top_k=args.top_k)
        t2 = time.perf_counter()

        embed_times.append(t1 - t0)
        faiss_times.append(t2 - t1)
        end_to_end_times.append(t2 - t0)

    embed_arr = np.array(embed_times, dtype=np.float64)
    faiss_arr = np.array(faiss_times, dtype=np.float64)
    e2e_arr = np.array(end_to_end_times, dtype=np.float64)

    print("queries", int(len(user_ids)))
    print("embedding", _percentiles_ms(embed_arr))
    print("faiss", _percentiles_ms(faiss_arr))
    print("end_to_end", _percentiles_ms(e2e_arr))


if __name__ == "__main__":
    main()
