"""items.csv'yi tek gecişte kompakt token artifact'ine cevirir (RAM-guvenli).

Her item icin: title/brand/category/attr-value tokenlarinin crc32 hash'leri
(uint32) flat dizide + offset dizileri; gender/age int8 kod. Toplam ~300 MB disk,
yukleme sonrasi RAM ~250 MB. 963k Python set'i tutmaktan (~1+ GB) kacinir.

Cikti: artifacts/item_tokens.npz
"""
import re
import time
import zlib

import numpy as np
import pandas as pd

import config as C
from data_utils import fold_tr, parse_attributes

TOKEN_RE = re.compile(r"[a-z0-9]+")
GENDER_CODE = {"kadin": 1, "erkek": 2, "unisex": 3}
AGE_CODE = {"cocuk": 1, "bebek": 2, "yetiskin": 3, "genc": 4}


def tok_hashes(text):
    if not text:
        return []
    return [zlib.crc32(t.encode()) for t in set(TOKEN_RE.findall(text))]


def main():
    t0 = time.time()
    fields = {"title": [], "brand": [], "cat": [], "attr": []}
    offsets = {k: [0] for k in fields}
    genders, ages = [], []
    n_rows = 0

    reader = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                         chunksize=100_000,
                         usecols=["item_id", "title", "category", "brand",
                                  "gender", "age_group", "attributes"])
    for chunk in reader:
        for r in chunk.itertuples(index=False):
            title = fold_tr(r.title.lower())
            brand = fold_tr(r.brand.lower())
            cat = fold_tr(r.category.lower())
            attrs = parse_attributes(fold_tr(r.attributes.lower()))
            a_text = " ".join(attrs.values())

            for key, txt in (("title", title), ("brand", brand),
                             ("cat", cat), ("attr", a_text)):
                h = tok_hashes(txt)
                fields[key].extend(h)
                offsets[key].append(offsets[key][-1] + len(h))
            genders.append(GENDER_CODE.get(fold_tr(r.gender.lower()), 0))
            ages.append(AGE_CODE.get(fold_tr(r.age_group.lower()), 0))
        n_rows += len(chunk)
        print(f"  {n_rows} item  ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for k in fields:
        out[f"{k}_vals"] = np.asarray(fields[k], dtype=np.uint32)
        out[f"{k}_off"] = np.asarray(offsets[k], dtype=np.int64)
    out["gender"] = np.asarray(genders, dtype=np.int8)
    out["age"] = np.asarray(ages, dtype=np.int8)
    np.savez(C.ARTIFACTS_DIR / "item_tokens.npz", **out)
    print(f"kaydedildi: item_tokens.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
