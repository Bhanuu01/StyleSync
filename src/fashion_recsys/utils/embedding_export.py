from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf


def export_item_embeddings(
    item_tower: tf.keras.Model,
    item_dataset: tf.data.Dataset,
    output_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    ids = []
    vecs = []
    for batch in item_dataset:
        item_id = batch["item_id"].numpy().astype(str)
        emb = item_tower(batch).numpy().astype(np.float32)
        ids.append(item_id)
        vecs.append(emb)

    ids_arr = np.concatenate(ids, axis=0)
    vecs_arr = np.concatenate(vecs, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, item_id=ids_arr, embedding=vecs_arr)
    return ids_arr, vecs_arr


def export_user_embedding(user_tower: tf.keras.Model, features: Dict[str, object]) -> np.ndarray:
    expected_specs = None
    try:
        args, kwargs = user_tower.structured_input_signature
        if args and isinstance(args[0], dict):
            expected_specs = args[0]
        elif kwargs and isinstance(kwargs, dict):
            expected_specs = kwargs
    except Exception:
        expected_specs = None

    merged_features: Dict[str, object] = dict(features)
    if expected_specs:
        for k, spec in expected_specs.items():
            if k in merged_features:
                continue
            if getattr(spec, "dtype", None) == tf.string:
                merged_features[k] = ""
            elif getattr(spec, "dtype", None) == tf.bool:
                merged_features[k] = False
            else:
                merged_features[k] = 0.0

    tf_features = {}
    for k, v in merged_features.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            tf_features[k] = tf.convert_to_tensor(v)
        else:
            tf_features[k] = tf.convert_to_tensor([v])

    try:
        emb = user_tower(tf_features).numpy().astype(np.float32)
        return emb
    except ValueError as e:
        msg = str(e)
        expected_keys = set(re.findall(r"'([A-Za-z0-9_]+)': TensorSpec", msg))
        missing = expected_keys.difference(tf_features.keys())
        if not missing:
            raise

        filled = dict(tf_features)
        for k in missing:
            m = re.search(r"'" + re.escape(k) + r"': TensorSpec\(shape=.*?dtype=tf\.([A-Za-z0-9_]+)", msg)
            dtype_name = m.group(1) if m else None
            dtype = getattr(tf, dtype_name, None) if dtype_name else None

            if dtype == tf.string:
                filled[k] = tf.constant([""], dtype=tf.string)
            elif dtype == tf.bool:
                filled[k] = tf.constant([False], dtype=tf.bool)
            elif dtype in {tf.int32, tf.int64, tf.uint32, tf.uint64}:
                filled[k] = tf.constant([0], dtype=dtype)
            elif dtype in {tf.float16, tf.float32, tf.float64}:
                filled[k] = tf.constant([0.0], dtype=dtype)
            else:
                filled[k] = tf.convert_to_tensor([0.0])

        emb = user_tower(filled).numpy().astype(np.float32)
        return emb
