from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass(frozen=True)
class HmFrames:
    interactions: pd.DataFrame
    customers: Optional[pd.DataFrame]
    articles: pd.DataFrame


def _read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, nrows=nrows)


def load_hm_frames(data_dir: str | Path, interactions_nrows: Optional[int] = None) -> HmFrames:
    data_dir = Path(data_dir)
    interactions = _read_csv(data_dir / "transactions_train.csv", nrows=interactions_nrows)
    articles = _read_csv(data_dir / "articles.csv")
    customers_path = data_dir / "customers.csv"
    customers = _read_csv(customers_path) if customers_path.exists() else None

    interactions = interactions.rename(columns={"customer_id": "user_id", "article_id": "item_id"})
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)

    articles = articles.rename(columns={"article_id": "item_id"})
    articles["item_id"] = articles["item_id"].astype(str)

    if "detail_desc" in articles.columns:
        articles["item_text"] = articles["detail_desc"].fillna("").astype(str)
    elif "prod_name" in articles.columns:
        articles["item_text"] = articles["prod_name"].fillna("").astype(str)
    else:
        articles["item_text"] = ""

    if customers is not None:
        customers = customers.rename(columns={"customer_id": "user_id"})
        customers["user_id"] = customers["user_id"].astype(str)

    return HmFrames(interactions=interactions, customers=customers, articles=articles)


def load_hm_articles(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    articles = _read_csv(data_dir / "articles.csv")

    articles = articles.rename(columns={"article_id": "item_id"})
    articles["item_id"] = articles["item_id"].astype(str)

    if "detail_desc" in articles.columns:
        articles["item_text"] = articles["detail_desc"].fillna("").astype(str)
    elif "prod_name" in articles.columns:
        articles["item_text"] = articles["prod_name"].fillna("").astype(str)
    else:
        articles["item_text"] = ""

    return articles[["item_id", "item_text"]].drop_duplicates("item_id").reset_index(drop=True)


def _make_image_table(image_embeddings: Tuple[np.ndarray, np.ndarray]):
    emb_item_ids, emb = image_embeddings
    emb_item_ids = emb_item_ids.astype(str)
    emb = emb.astype(np.float32)

    lookup = tf.keras.layers.StringLookup(vocabulary=emb_item_ids, mask_token=None, oov_token="[OOV]")
    emb_matrix = tf.constant(emb, dtype=tf.float32)
    dim = int(emb.shape[1])

    def lookup_fn(item_ids: tf.Tensor) -> tf.Tensor:
        idx = lookup(item_ids)
        idx0 = tf.maximum(idx - 1, 0)
        gathered = tf.gather(emb_matrix, idx0)
        is_oov = tf.equal(idx, 0)
        zeros = tf.zeros([dim], dtype=tf.float32)
        return tf.where(tf.expand_dims(is_oov, -1), zeros, gathered)

    return lookup_fn


def build_tf_datasets(
    frames: HmFrames,
    batch_size: int,
    image_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    seed: int = 42,
    train_fraction: float = 0.9,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    interactions = frames.interactions[["user_id", "item_id"]].copy()

    if frames.customers is not None and "age" in frames.customers.columns:
        age_map = frames.customers[["user_id", "age"]].copy()
        age_map["age"] = pd.to_numeric(age_map["age"], errors="coerce").fillna(0).astype(np.float32)
        interactions = interactions.merge(age_map, on="user_id", how="left")
        interactions["age"] = interactions["age"].fillna(0).astype(np.float32)

    items = frames.articles[["item_id", "item_text"]].drop_duplicates("item_id").copy()
    interactions = interactions.merge(items, on="item_id", how="left")
    interactions["item_text"] = interactions["item_text"].fillna("").astype(str)

    image_lookup = _make_image_table(image_embeddings) if image_embeddings is not None else None

    interactions = interactions.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split = int(len(interactions) * train_fraction)
    train_df = interactions.iloc[:split]
    test_df = interactions.iloc[split:]

    def df_to_ds(df: pd.DataFrame) -> tf.data.Dataset:
        features = {
            "user_id": df["user_id"].astype(str).to_numpy(),
            "item_id": df["item_id"].astype(str).to_numpy(),
            "item_text": df["item_text"].astype(str).to_numpy(),
        }
        if "age" in df.columns:
            features["age"] = df["age"].astype(np.float32).to_numpy()
        ds = tf.data.Dataset.from_tensor_slices(features)
        return ds

    train = df_to_ds(train_df).shuffle(100_000, seed=seed, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test = df_to_ds(test_df).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if image_lookup is not None:
        train = train.map(lambda x: {**x, "image_embedding": image_lookup(x["item_id"])}, num_parallel_calls=tf.data.AUTOTUNE)
        test = test.map(lambda x: {**x, "image_embedding": image_lookup(x["item_id"])}, num_parallel_calls=tf.data.AUTOTUNE)

    item_ds = tf.data.Dataset.from_tensor_slices(
        {
            "item_id": items["item_id"].astype(str).to_numpy(),
            "item_text": items["item_text"].astype(str).to_numpy(),
        }
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if image_lookup is not None:
        item_ds = item_ds.map(lambda x: {**x, "image_embedding": image_lookup(x["item_id"])}, num_parallel_calls=tf.data.AUTOTUNE)

    return train, test, item_ds


def get_vocabularies(frames: HmFrames) -> Tuple[np.ndarray, np.ndarray]:
    user_ids = frames.interactions["user_id"].astype(str).unique()
    item_ids = frames.articles["item_id"].astype(str).unique()
    return user_ids, item_ids


def load_image_embeddings_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    item_id = data["item_id"].astype(str)
    embedding = data["embedding"].astype(np.float32)
    return item_id, embedding


def build_item_dataset(
    articles: pd.DataFrame,
    batch_size: int,
    image_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> tf.data.Dataset:
    items = articles[["item_id", "item_text"]].drop_duplicates("item_id").copy()

    image_lookup = _make_image_table(image_embeddings) if image_embeddings is not None else None

    ds = tf.data.Dataset.from_tensor_slices(
        {
            "item_id": items["item_id"].astype(str).to_numpy(),
            "item_text": items["item_text"].astype(str).to_numpy(),
        }
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if image_lookup is not None:
        ds = ds.map(lambda x: {**x, "image_embedding": image_lookup(x["item_id"])}, num_parallel_calls=tf.data.AUTOTUNE)

    return ds
