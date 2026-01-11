from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _require_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:
        raise ImportError("faiss is not installed. Install requirements-faiss.txt") from e


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


@dataclass(frozen=True)
class FaissArtifacts:
    index_path: Path
    ids_path: Path


def build_ip_index(
    item_ids: np.ndarray,
    item_embeddings: np.ndarray,
    output_dir: str | Path,
    normalize: bool = True,
) -> FaissArtifacts:
    faiss = _require_faiss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xb = item_embeddings.astype(np.float32)
    if normalize:
        xb = l2_normalize(xb)

    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    index_path = output_dir / "faiss.index"
    ids_path = output_dir / "item_ids.npy"

    faiss.write_index(index, str(index_path))
    np.save(ids_path, item_ids)

    return FaissArtifacts(index_path=index_path, ids_path=ids_path)


def load_ip_index(artifacts: FaissArtifacts):
    faiss = _require_faiss()
    index = faiss.read_index(str(artifacts.index_path))
    item_ids = np.load(artifacts.ids_path, allow_pickle=False)
    return index, item_ids


def search(
    index,
    item_ids: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    q = query_embeddings.astype(np.float32)
    if normalize:
        q = l2_normalize(q)
    scores, idx = index.search(q, top_k)
    rec_ids = item_ids[idx]
    return rec_ids, scores
