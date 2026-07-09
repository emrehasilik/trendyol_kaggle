"""v6 hazirlik: katalog + termler icin YENI govdenin (bge-reranker-v2-m3) token cache'i.

tokenize_ce_cache.py'nin v6 karsiligi; farklar:
- Model TK2_MODEL env ile secilir (vars. BAAI/bge-reranker-v2-m3). Cache dosya
  adina model slug'i gomulur -> farkli tokenizer'larin cache'i karisamaz.
- Metin ORIJINAL harf durumuyla birakilir (XLM-R/mDeBERTa cased; on-egitim
  dagilimina en yakin girdi ham metin). BERTurk-uncased'in lowercase kurali
  burada GECERSIZ.
- Item sablonu zenginlestirildi: tam kategori yolu (eskiden son 2 segment),
  attribute tavani 8->10 k:v, item token payi 110->160 (CE_V6_MAX_ITEM_TOKENS).

Cikti: artifacts/ce_tokens_terms_v6_{slug}.npz, artifacts/ce_tokens_items_v6_{slug}.npz
"""
import re
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import config as C
from data_utils import parse_attributes
from text_builder import ATTR_WHITELIST

_WL_SET = set(ATTR_WHITELIST)
# Python .lower() Turkce I/İ'yi yanlis esler; whitelist eslesmesi icin duzeltme
_TR_I_MAP = str.maketrans({"İ": "i", "I": "ı"})


def model_slug(name: str = None) -> str:
    name = name or C.CE_V6_MODEL_NAME
    return re.sub(r"[^A-Za-z0-9_-]", "-", name.split("/")[-1])


def tokens_path(kind: str):
    """kind: 'terms' | 'items'"""
    return C.ARTIFACTS_DIR / f"ce_tokens_{kind}_v6_{model_slug()}.npz"


def item_text_v6(title, brand, category, gender, age_group, attributes) -> str:
    parts = [title.strip()]
    if brand and brand.lower() != "unknown":
        parts.append(brand.strip())
    if category:
        segs = [s.strip() for s in category.split("/") if s.strip()]
        parts.append(" ".join(segs))  # tam yol (v1'de son 2 segmentti)
    ga = " ".join(v for v in (gender, age_group) if v and v.lower() != "unknown")
    if ga:
        parts.append(ga)
    attrs = parse_attributes(attributes.translate(_TR_I_MAP))
    kv = [f"{k}: {v}" for k, v in attrs.items() if k in _WL_SET and v]
    if kv:
        parts.append(", ".join(kv[:10]))
    return " | ".join(parts)


def _encode_to_npz(tok, texts, max_tokens, out_path, t0, label):
    enc = tok(texts, add_special_tokens=False, truncation=True,
              max_length=max_tokens)["input_ids"]
    vals = np.asarray([i for ids in enc for i in ids], dtype=np.uint32)
    off = np.cumsum([0] + [len(ids) for ids in enc]).astype(np.int64)
    np.savez(out_path, vals=vals, off=off)
    print(f"{label}: {len(texts)} metin, ort {vals.size/len(texts):.1f} token "
          f"({time.time()-t0:.0f}s) -> {out_path.name}")


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(C.CE_V6_MODEL_NAME, cache_dir=C.HF_CACHE)
    print(f"tokenizer hazir: {C.CE_V6_MODEL_NAME}  vocab={len(tok)}")

    tp = tokens_path("terms")
    if tp.exists():
        print(f"{tp.name} mevcut, atlaniyor")
    else:
        terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
        _encode_to_npz(tok, list(terms["query"]), C.CE_V6_MAX_TERM_TOKENS,
                       tp, t0, "termler")

    ip = tokens_path("items")
    if ip.exists():
        print(f"{ip.name} mevcut, atlaniyor")
        return
    all_vals, lens = [], []
    n_rows = 0
    reader = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                         chunksize=50_000,
                         usecols=["item_id", "title", "category", "brand",
                                  "gender", "age_group", "attributes"])
    for chunk in reader:
        texts = [item_text_v6(r.title, r.brand, r.category, r.gender,
                              r.age_group, r.attributes)
                 for r in chunk.itertuples(index=False)]
        enc = tok(texts, add_special_tokens=False, truncation=True,
                  max_length=C.CE_V6_MAX_ITEM_TOKENS)["input_ids"]
        for ids in enc:
            all_vals.append(np.asarray(ids, dtype=np.uint32))
            lens.append(len(ids))
        n_rows += len(chunk)
        print(f"  {n_rows} item  ({time.time()-t0:.0f}s)", flush=True)

    vals = np.concatenate(all_vals)
    off = np.cumsum([0] + lens).astype(np.int64)
    np.savez(ip, vals=vals, off=off)
    print(f"kaydedildi: {ip.name}  toplam {time.time()-t0:.0f}s  "
          f"ort item token={vals.size/n_rows:.1f}")


if __name__ == "__main__":
    main()
