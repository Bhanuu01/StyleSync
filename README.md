# StyleSync (Two-Tower Retrieval Recommender)

## Live demo

- **Streamlit app**: https://stylesync-88c6xauzznaxkddpwnwrhn.streamlit.app
- **Backend (FastAPI on Render)**: https://stylesync-ltjf.onrender.com
  - **Swagger**: https://stylesync-ltjf.onrender.com/docs
  - **POST** `https://stylesync-ltjf.onrender.com/recommend`

## What this repo contains

- Two-tower retrieval model implemented with TensorFlow Recommenders (TFRS)
- Data loader utilities for the H&M dataset (Kaggle)
- Utilities to export item embeddings and build/query a FAISS index for candidate generation

## Expected dataset layout

Point `--data-dir` at a folder containing (at minimum):

- `transactions_train.csv`
- `articles.csv`

Optionally:

- `customers.csv`

## Install

Create a virtualenv, then:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is intentionally **frontend-only** (Streamlit Cloud uses it).
- For **training/FAISS build** locally, also install:

```bash
pip install -r requirements-ml.txt
pip install -r requirements-faiss.txt
pip install -e .
```

- For **backend serving** locally, also install:

```bash
pip install -r requirements-backend.txt
```

If you previously installed TensorFlow 2.16+ (Keras 3), reinstall after pinning dependencies in `requirements-ml.txt`.

## Run locally

Backend:

```bash
PYTHONPATH=src .venv/bin/uvicorn serving.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/frontend/app.py
```

## Deploy (Render backend + Streamlit Cloud frontend)

### 1) Deploy backend (Render)

- Build command:

```bash
pip install -r requirements-backend.txt
```

- Start command:

```bash
PYTHONPATH=src python -m uvicorn serving.main:app --host 0.0.0.0 --port $PORT
```

Important: the backend requires these artifacts to exist in the deployed environment:

- `artifacts/faiss/faiss.index`
- `artifacts/faiss/item_ids.npy`
- `artifacts/model/user_tower/` (SavedModel)

### 2) Deploy frontend (Streamlit Community Cloud)

- App entry point: `src/frontend/app.py`
- Set Streamlit **Secrets**:

```toml
BACKEND_URL = "https://stylesync-ltjf.onrender.com/recommend"
```

## Train the retrieval model

```bash
python scripts/train_retrieval.py --data-dir /path/to/hm
```

If you run out of memory on a laptop, train on a subset of transactions:

```bash
python scripts/train_retrieval.py --data-dir /path/to/hm --transactions-nrows 500000
```

Artifacts saved to:

- `artifacts/retrieval_model/`

This directory will contain:

- `artifacts/retrieval_model/user_tower/`
- `artifacts/retrieval_model/item_tower/`

## Export item embeddings

```bash
python scripts/export_embeddings.py --data-dir /path/to/hm
```

Artifacts saved to:

- `artifacts/item_embeddings.npz`

## Build FAISS index

```bash
python scripts/build_faiss.py --embeddings artifacts/item_embeddings.npz
```

Artifacts saved to:

- `artifacts/faiss/faiss.index`
- `artifacts/faiss/item_ids.npy`

## Query FAISS for top-K candidates

```bash
python scripts/query_faiss.py --user-id <customer_id> --top-k 20
```

## Image embeddings (ResNet50)

If you have the H&M images folder available (typically `images/` under the dataset directory), you can precompute image embeddings:

```bash
python scripts/embed_images_resnet50.py --data-dir /path/to/hm
```

This will write:

- `artifacts/image_embeddings_resnet50.npz`

Then train / export with images:

```bash
python scripts/train_retrieval.py --data-dir /path/to/hm --image-embeddings artifacts/image_embeddings_resnet50.npz
python scripts/export_embeddings.py --data-dir /path/to/hm --image-embeddings artifacts/image_embeddings_resnet50.npz
```

If you want to include age (only if you have `customers.csv`):

```bash
python scripts/query_faiss.py --user-id <customer_id> --age 28 --top-k 20
```
