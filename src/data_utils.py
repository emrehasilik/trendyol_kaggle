"""Veri yükleme ve temel temizleme yardımcı fonksiyonları."""
import re
import pandas as pd

# Veride bazi title'lar Turkce karakterli ("sampuan"), bazilari ASCII'ye
# duzlestirilmis ("sampuan") yaziliyor. Ayni kelime iki farkli token olarak
# gorunup TF-IDF eslesmesini kirdigi icin tum metni tek bir forma (ASCII)
# indirgiyoruz - query ve title artik hep ayni alfabede karsilastiriliyor.
_TR_FOLD_MAP = str.maketrans({
    "ş": "s", "Ş": "s", "ç": "c", "Ç": "c", "ı": "i", "İ": "i",
    "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})


def fold_tr(text: str) -> str:
    if not text:
        return ""
    return text.translate(_TR_FOLD_MAP)


def load_items(path="items.csv"):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["title_raw"] = df["title"].fillna("").str.lower()  # embedding modeli icin Turkce karakterler korunur
    for col in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        df[col] = df[col].fillna("").str.lower().map(fold_tr)
    return df


def load_terms(path="terms.csv"):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["query_raw"] = df["query"].fillna("").str.lower()  # embedding modeli icin Turkce karakterler korunur
    df["query"] = df["query"].fillna("").str.lower().map(fold_tr)
    return df


def load_training_pairs(path="training_pairs.csv"):
    return pd.read_csv(path, dtype={"id": str, "term_id": str, "item_id": str, "label": int})


def load_submission_pairs(path="submission_pairs.csv"):
    return pd.read_csv(path, dtype=str)


_ATTR_SPLIT_RE = re.compile(r",\s*(?=[^,]+?:)")


def parse_attributes(attr_str):
    """'anahtar: deger, anahtar2: deger2' -> dict. Bozuk/boş girişlere toleranslı."""
    result = {}
    if not attr_str:
        return result
    for part in _ATTR_SPLIT_RE.split(attr_str):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        k = k.strip().lower()
        v = v.strip().lower()
        if k:
            result[k] = v
    return result
