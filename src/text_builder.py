"""Faz 3: cross-encoder input metni.

BERTurk uncased kendi lowercase'ini yapar; ASCII folding UYGULANMAZ
(model dogal Turkce metinle egitildi). Sablon:
  item : title | brand | son 2 kategori segmenti | gender age | k:v attributeler
Sorgu tarafi ayrica verilir (tokenizer text_pair olarak birlestirir).
"""
import pandas as pd

import config as C
from data_utils import parse_attributes

# 150k item taramasindaki en sik + ayirt edici anahtarlar
ATTR_WHITELIST = [
    "renk", "materyal", "desen", "kumaş tipi", "kalıp", "yaka tipi",
    "kol tipi", "sezon", "boy", "koleksiyon", "beden", "kapasite", "form",
]
_WL_SET = set(ATTR_WHITELIST)


def item_text(title, brand, category, gender, age_group, attributes) -> str:
    parts = [title.strip()]
    if brand and brand != "unknown":
        parts.append(brand.strip())
    if category:
        segs = [s.strip() for s in category.split("/") if s.strip()]
        parts.append(" ".join(segs[-2:]))
    ga = " ".join(v for v in (gender, age_group) if v and v != "unknown")
    if ga:
        parts.append(ga)
    attrs = parse_attributes(attributes)
    kv = [f"{k}: {v}" for k, v in attrs.items() if k in _WL_SET and v]
    if kv:
        parts.append(", ".join(kv[:8]))
    return " | ".join(parts)


def build_item_texts() -> pd.DataFrame:
    """Tum katalog icin item_id -> metin. Lowercase, Turkce karakterler korunur."""
    items = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False)
    for col in ["title", "brand", "category", "gender", "age_group", "attributes"]:
        items[col] = items[col].str.lower()
    texts = [
        item_text(r.title, r.brand, r.category, r.gender, r.age_group, r.attributes)
        for r in items.itertuples(index=False)
    ]
    return pd.DataFrame({"item_id": items["item_id"].values, "text": texts})


def build_term_texts() -> pd.DataFrame:
    terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame({"term_id": terms["term_id"].values,
                         "text": terms["query"].str.lower().values})
