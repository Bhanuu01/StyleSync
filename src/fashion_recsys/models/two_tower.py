from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs


@tf.keras.utils.register_keras_serializable()
class UserTower(tf.keras.Model):
    def __init__(self, user_vocab, embedding_dim: int):
        super().__init__()
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_embedding = tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), embedding_dim)
        self.age_norm = tf.keras.layers.Normalization(axis=None)
        self.age_dense = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.proj = tf.keras.layers.Dense(embedding_dim, activation=None)

    def adapt(self, age_ds: tf.data.Dataset) -> None:
        self.age_norm.adapt(age_ds)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_id = inputs["user_id"]
        user_vec = self.user_embedding(self.user_lookup(user_id))
        if "age" in inputs:
            age = tf.cast(inputs["age"], tf.float32)
            age_vec = self.age_dense(tf.expand_dims(self.age_norm(age), -1))
            return self.proj(tf.concat([user_vec, age_vec], axis=-1))
        return self.proj(user_vec)


@tf.keras.utils.register_keras_serializable()
class ItemTower(tf.keras.Model):
    def __init__(
        self,
        item_vocab,
        embedding_dim: int,
        max_tokens: int = 20_000,
        text_seq_len: int = 32,
    ):
        super().__init__()
        self.item_lookup = tf.keras.layers.StringLookup(vocabulary=item_vocab, mask_token=None)
        self.item_embedding = tf.keras.layers.Embedding(self.item_lookup.vocabulary_size(), embedding_dim)

        self.text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=text_seq_len,
        )
        self.text_embedding = tf.keras.layers.Embedding(max_tokens, embedding_dim)
        self.text_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.image_dense = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.proj = tf.keras.layers.Dense(embedding_dim, activation=None)

    def adapt(self, text_ds: tf.data.Dataset) -> None:
        self.text_vectorizer.adapt(text_ds)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        item_id = inputs["item_id"]
        item_vec = self.item_embedding(self.item_lookup(item_id))

        item_text = inputs.get("item_text", tf.fill(tf.shape(item_id), ""))
        text_tokens = self.text_vectorizer(item_text)
        text_vec = self.text_pool(self.text_embedding(text_tokens))

        parts = [item_vec, text_vec]

        if "image_embedding" in inputs:
            img = tf.cast(inputs["image_embedding"], tf.float32)
            parts.append(self.image_dense(img))

        return self.proj(tf.concat(parts, axis=-1))


@tf.keras.utils.register_keras_serializable()
class TwoTowerRetrievalModel(tfrs.models.Model):
    def __init__(
        self,
        user_tower: UserTower,
        item_tower: ItemTower,
        candidate_dataset: tf.data.Dataset,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

        metrics = tfrs.metrics.FactorizedTopK(candidates=candidate_dataset.map(self.item_tower))
        self.task = tfrs.tasks.Retrieval(metrics=metrics, temperature=temperature)

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        user_inputs = {"user_id": features["user_id"]}
        if "age" in features:
            user_inputs["age"] = features["age"]

        item_inputs = {"item_id": features["item_id"]}
        if "item_text" in features:
            item_inputs["item_text"] = features["item_text"]
        if "image_embedding" in features:
            item_inputs["image_embedding"] = features["image_embedding"]

        user_embeddings = self.user_tower(user_inputs)
        item_embeddings = self.item_tower(item_inputs)
        return self.task(user_embeddings, item_embeddings, compute_metrics=not training)
