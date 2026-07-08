"""Faz 3 hazirlik: tum katalog + termler icin CE token cache'i.

Her unique item metni ve term sorgusu BIR KEZ tokenize edilir; egitim/inference
sirasinda sadece id dizileri birlestirilir (Windows'ta CPU tokenizasyon darbogazini
kaldirir). items.csv streaming okunur (RAM-guvenli).

Cikti: artifacts/ce_tokens_items.npz (vals uint32 flat + off int64)
       artifacts/ce_tokens_terms.npz
"""
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import config as C
from text_builder import item_text

MAX_ITEM_TOKENS = 110   # max_len 128 - sorgu payi; item tarafi burada kirpilir
MAX_TERM_TOKENS = 32

# "python tokenize_ce_cache.py mdeberta" -> mDeBERTa tokenizer'iyla ayri cache
VARIANT = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
MODEL_NAMES = {"": C.CE_MODEL_NAME, "mdeberta": "microsoft/mdeberta-v3-base",
               "xlmr": "FacebookAI/xlm-roberta-base"}
MODEL_NAME = MODEL_NAMES[VARIANT]
SUF = f"_{VARIANT}" if VARIANT else ""


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=C.HF_CACHE)
    print(f"tokenizer hazir: {MODEL_NAME}  vocab={tok.vocab_size}")

    # ---- termler (kucuk, tek seferde)
    terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
    enc = tok(list(terms["query"].str.lower()), add_special_tokens=False,
              truncation=True, max_length=MAX_TERM_TOKENS)["input_ids"]
    vals = np.asarray([i for ids in enc for i in ids], dtype=np.uint32)
    off = np.cumsum([0] + [len(ids) for ids in enc]).astype(np.int64)
    np.savez(C.ARTIFACTS_DIR / f"ce_tokens_terms{SUF}.npz", vals=vals, off=off)
    print(f"termler tokenize edildi: {len(terms)}  ({time.time()-t0:.0f}s)")

    # ---- itemlar (streaming)
    all_vals, lens = [], []
    n_rows = 0
    reader = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                         chunksize=50_000,
                         usecols=["item_id", "title", "category", "brand",
                                  "gender", "age_group", "attributes"])
    for chunk in reader:
        for col in ["title", "brand", "category", "gender", "age_group", "attributes"]:
            chunk[col] = chunk[col].str.lower()
        texts = [item_text(r.title, r.brand, r.category, r.gender, r.age_group,
                           r.attributes) for r in chunk.itertuples(index=False)]
        enc = tok(texts, add_special_tokens=False, truncation=True,
                  max_length=MAX_ITEM_TOKENS)["input_ids"]
        for ids in enc:
            all_vals.append(np.asarray(ids, dtype=np.uint32))
            lens.append(len(ids))
        n_rows += len(chunk)
        print(f"  {n_rows} item  ({time.time()-t0:.0f}s)", flush=True)

    vals = np.concatenate(all_vals)
    off = np.cumsum([0] + lens).astype(np.int64)
    np.savez(C.ARTIFACTS_DIR / f"ce_tokens_items{SUF}.npz", vals=vals, off=off)
    print(f"kaydedildi: ce_tokens_items{SUF}.npz  toplam {time.time()-t0:.0f}s  "
          f"ort item token={vals.size/n_rows:.1f}")


if __name__ == "__main__":
    main()
