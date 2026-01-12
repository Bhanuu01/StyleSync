from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/recommend"


def _get_backend_url() -> str:
    try:
        if "BACKEND_URL" in st.secrets:
            return str(st.secrets["BACKEND_URL"]).strip()
    except Exception:
        pass

    return str(os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)).strip()


BACKEND_URL = _get_backend_url()
PLACEHOLDER_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"


def _normalize_article_id(x: str) -> str:
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(10)
    return s


def _image_path(images_dir: Path, article_id_norm: str) -> Path | None:
    if not images_dir.exists():
        return None

    if not article_id_norm or not article_id_norm.isdigit():
        return None

    p = images_dir / article_id_norm[:3] / f"{article_id_norm}.jpg"
    if p.exists():
        return p
    return None


@st.cache_data
def load_thumbnail(image_path: str, size: int = 256) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    thumb = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return thumb


@st.cache_data
def load_articles(articles_csv: str) -> pd.DataFrame:
    usecols = [
        "article_id",
        "prod_name",
        "product_type_name",
        "detail_desc",
        "graphical_appearance_name",
    ]
    df = pd.read_csv(articles_csv, usecols=usecols, dtype={"article_id": "string"})
    df["article_id"] = df["article_id"].astype(str)
    df["article_id_norm"] = df["article_id"].map(_normalize_article_id)
    return df


@st.cache_data
def load_sample_customer_ids(
    data_dir: str,
    nrows: int = 200_000,
    n_sample: int = 50,
    seed: int = 42,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    data_dir_p = Path(data_dir)
    customers_csv = data_dir_p / "customers.csv"
    tx_csv = data_dir_p / "transactions_train.csv"

    src = customers_csv if customers_csv.exists() else tx_csv
    if not src.exists():
        return []

    df = pd.read_csv(src, usecols=["customer_id"], nrows=nrows, dtype={"customer_id": "string"})
    ids = df["customer_id"].astype(str).dropna().drop_duplicates().tolist()
    if not ids:
        return []

    if exclude_ids:
        exclude = set(str(x) for x in exclude_ids)
        ids = [x for x in ids if x not in exclude]
        if not ids:
            return []

    if len(ids) <= n_sample:
        return ids
    return pd.Series(ids).sample(n=n_sample, random_state=seed).tolist()


def main() -> None:
    st.set_page_config(layout="wide", page_title="StyleSync")
    st.title("StyleSync")

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] { background: #0b1220; border: 1px solid #1f2a44; padding: 12px 16px; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    repo_root = Path(__file__).resolve().parents[2]
    articles_csv = repo_root / "data" / "hm" / "articles.csv"
    images_dir = repo_root / "data" / "hm" / "images"
    data_dir = repo_root / "data" / "hm"

    if not articles_csv.exists():
        st.error(f"Could not find articles.csv at: {articles_csv}")
        st.stop()

    articles_df = load_articles(str(articles_csv))

    with st.sidebar:
        st.header("Controls")

        featured_users = {
            "High Volume Shopper": "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
            "Young Adult": "00007d2de826758b65a93dd24ce629ed66842531df6699338c55742e020a3e77",
            "Senior Customer": "00083cda041544b2fbb0e0d2905ad17da7cf1007526fb4c73235dccbbc132280",
        }

        extra_n = st.slider("Extra sample users", min_value=0, max_value=100, value=25, step=5)
        sampled = (
            load_sample_customer_ids(
                str(data_dir),
                n_sample=int(extra_n),
                exclude_ids=list(featured_users.values()),
            )
            if extra_n > 0
            else []
        )
        if extra_n > 0 and not sampled:
            st.caption(
                "Extra sample users are unavailable because `customers.csv`/`transactions_train.csv` is not present in `data/hm/`."
            )
        elif extra_n > 0 and len(sampled) < int(extra_n):
            st.caption(f"Only {len(sampled)} sample user(s) available from the local dataset.")

        user_options: dict[str, str] = dict(featured_users)
        for i, cid in enumerate(sampled):
            label = f"Sample User {i+1:02d}"
            if cid not in user_options.values():
                user_options[label] = cid
        user_options["Custom (paste id)"] = ""

        selected_profile = st.selectbox("User", options=list(user_options.keys()))
        user_id = user_options[selected_profile]
        if selected_profile == "Custom (paste id)":
            user_id = st.text_input("Customer ID", value=featured_users["High Volume Shopper"]).strip()

        top_k = st.slider("Top K", min_value=1, max_value=50, value=10, step=1)
        run = st.button("Get Recommendations")

    if not run:
        st.info("Enter a Customer ID and click **Get Recommendations**.")
        return

    resp = None
    last_err: Exception | None = None
    payload = {"user_id": user_id.strip(), "top_k": int(top_k)}
    timeouts_s = [15, 45, 90]
    for attempt, timeout_s in enumerate(timeouts_s, start=1):
        try:
            if attempt > 1:
                st.info("Backend is waking up… retrying request.")
            resp = requests.post(BACKEND_URL, json=payload, timeout=timeout_s)
            last_err = None
            break
        except requests.RequestException as e:
            last_err = e

    if resp is None:
        st.error(
            "Backend request failed. If you just deployed the backend, it may be sleeping and needs a minute to wake up. "
            "Please try again.\n\n"
            f"Backend URL: `{BACKEND_URL}`\n\nError: {last_err}"
        )
        return

    if resp.status_code != 200:
        st.error(f"Backend returned HTTP {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    item_ids = [str(x) for x in data.get("item_ids", [])]
    scores = [float(x) for x in data.get("scores", [])]
    item_ids_norm = [_normalize_article_id(x) for x in item_ids]
    latency_ms = float(data.get("latency_ms", 0.0))

    st.metric("Latency (ms)", f"{latency_ms:.3f}")

    if not item_ids:
        st.warning("No recommendations returned.")
        return

    if scores and len(scores) != len(item_ids):
        st.warning("Backend returned scores with a different length than item_ids; ignoring scores.")
        scores = []

    articles_lookup = articles_df.drop_duplicates(subset=["article_id_norm"], keep="first").set_index("article_id_norm")
    ordered = articles_lookup.reindex(item_ids_norm).reset_index()

    missing_ids = [
        str(ordered.loc[i, "article_id_norm"]) if pd.notna(ordered.loc[i, "article_id_norm"]) else item_ids_norm[i]
        for i in range(len(item_ids_norm))
        if pd.isna(ordered.loc[i, "prod_name"])
    ]
    if missing_ids:
        st.warning(
            "Some recommended item_ids were not found in articles.csv. "
            "This is usually an ID formatting mismatch. Showing available items.\n\n"
            f"Missing: {', '.join(missing_ids[:10])}{' ...' if len(missing_ids) > 10 else ''}"
        )

    st.subheader("Recommended Items")

    n_cols = 4
    rows = (len(item_ids_norm) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            i = r * n_cols + c
            if i >= len(item_ids_norm):
                continue

            rec_id_norm = item_ids_norm[i]
            row0 = ordered.iloc[i]
            with cols[c]:
                st.markdown(f"#### #{i+1}")

                img = _image_path(images_dir, rec_id_norm)
                if img is not None:
                    st.image(load_thumbnail(str(img), size=256), width=256)
                else:
                    st.image(PLACEHOLDER_IMAGE_URL, width=256)

                if pd.isna(row0.get("prod_name")):
                    st.markdown(f"**{rec_id_norm}**")
                    continue

                st.markdown(f"**{row0['prod_name']}**")
                if scores:
                    raw = float(scores[i])
                    match = max(0.0, min(100.0, (raw + 1.0) * 50.0))
                    st.caption(f"{match:.0f}% Match")
                st.caption(f"{row0['product_type_name']} · {row0['graphical_appearance_name']}")
                with st.expander("Details", expanded=False):
                    desc = str(row0.get("detail_desc") or "").strip()
                    st.write(desc if desc else "No description")


if __name__ == "__main__":
    main()
