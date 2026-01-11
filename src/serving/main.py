from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fashion_recsys.utils.faiss_index import FaissArtifacts, load_ip_index, search


class RecommendRequest(BaseModel):
    user_id: str
    age: Optional[float] = None
    top_k: int = Field(default=10, ge=1, le=1000)


class RecommendResponse(BaseModel):
    item_ids: list[str]
    scores: list[float]
    latency_ms: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_user_tower(serving_fn, *, user_id: str, age, age_dtype: tf.dtypes.DType) -> np.ndarray:
    if age_dtype in {tf.int32, tf.int64, tf.uint32, tf.uint64}:
        age_tensor = tf.constant([int(age)], dtype=age_dtype)
    else:
        age_tensor = tf.constant([float(age)], dtype=age_dtype)

    feats = {
        "user_id": tf.constant([user_id], dtype=tf.string),
        "age": age_tensor,
    }

    try:
        out = serving_fn(feats)
    except TypeError:
        out = serving_fn(**feats)

    if isinstance(out, dict):
        if "embedding" in out:
            emb = out["embedding"]
        else:
            emb = next(iter(out.values()))
    else:
        emb = out

    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    return np.asarray(emb, dtype=np.float32)


app = FastAPI(title="StyleSync")

_user_serving_fn = None
_faiss_index = None
_item_ids = None
_age_dtype = tf.float32
_rng = np.random.default_rng()


@app.on_event("startup")
def _startup() -> None:
    global _user_serving_fn, _faiss_index, _item_ids, _age_dtype

    root = _repo_root()

    faiss_dir = root / "artifacts" / "faiss"
    artifacts = FaissArtifacts(index_path=faiss_dir / "faiss.index", ids_path=faiss_dir / "item_ids.npy")
    if not artifacts.index_path.exists() or not artifacts.ids_path.exists():
        raise RuntimeError(f"FAISS artifacts not found under {faiss_dir}. Run scripts/build_faiss.py first.")

    _faiss_index, _item_ids = load_ip_index(artifacts)

    model_path = root / "artifacts" / "model" / "user_tower"
    if not model_path.exists():
        raise RuntimeError(
            f"User tower SavedModel not found at {model_path}. Run scripts/train_retrieval.py with --export-dir artifacts/model"
        )

    loaded = tf.saved_model.load(str(model_path))
    if hasattr(loaded, "signatures") and "serving_default" in loaded.signatures:
        _user_serving_fn = loaded.signatures["serving_default"]
    else:
        _user_serving_fn = loaded

    try:
        _args, kwargs = _user_serving_fn.structured_input_signature
        if isinstance(kwargs, dict) and "age" in kwargs:
            _age_dtype = kwargs["age"].dtype
    except Exception:
        _age_dtype = tf.float32


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if _user_serving_fn is None or _faiss_index is None or _item_ids is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    top_k = int(req.top_k)
    if top_k > int(len(_item_ids)):
        top_k = int(len(_item_ids))

    age = 0.0 if req.age is None else float(req.age)
    if age < 0.0 or age > 120.0:
        raise HTTPException(status_code=422, detail="age must be between 0 and 120")

    candidate_pool_size = min(int(len(_item_ids)), max(top_k, top_k * 3))

    t0 = time.perf_counter()
    emb = _run_user_tower(_user_serving_fn, user_id=req.user_id, age=age, age_dtype=_age_dtype)
    rec_ids, _scores = search(index=_faiss_index, item_ids=_item_ids, query_embeddings=emb, top_k=candidate_pool_size)

    pool_ids = np.asarray(rec_ids[0]).copy()
    pool_scores = np.asarray(_scores[0]).copy()

    perm = np.arange(pool_ids.shape[0])
    _rng.shuffle(perm)
    perm = perm[:top_k]

    pool_ids = pool_ids[perm]
    pool_scores = pool_scores[perm]
    t1 = time.perf_counter()

    ids = [str(x) for x in pool_ids.tolist()]
    scores = [float(x) for x in pool_scores.tolist()]
    return RecommendResponse(item_ids=ids, scores=scores, latency_ms=(t1 - t0) * 1000.0)
