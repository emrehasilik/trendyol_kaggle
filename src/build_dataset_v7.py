"""v7 dataseti: pseudo-label dongusunun YENI OGRETMENLE (v6, farkli model sinifi) turu.

DERS #3/#5 geregi eski (v2/v3 soylu) pseudo-etiketler ATILIR; capalar korunur,
pseudo-etiketler v6'nin test skorlarindan YENIDEN uretilir. v6 farkli sinif bir
model oldugundan gri-bolge hatalari farkli -> dongu gercekten yeni bilgi tasir.

- Capa (v5'ten): pos (gercek pozitif), annvet (CE-onayli hard neg), v1neg,
  proxy_pos (gercek), proxy_neg.
- Pseudo (v6'dan): skor > 0.95 -> pl_pos (terim basina 18), < 0.05 -> pl_neg (15)
  — v4'un uzlasma tarifiyle ayni tavanlar.

Girdi : artifacts/train_dataset_v5.parquet, artifacts/ce_v6_test.npy
Cikti : artifacts/train_dataset_v7.parquet
"""
import numpy as np
import pandas as pd

import config as C

ANCHOR_SOURCES = ["pos", "annvet", "v1neg", "proxy_pos", "proxy_neg"]
PL_POS_THR, PL_POS_PER_TERM = 0.95, 18
PL_NEG_THR, PL_NEG_PER_TERM = 0.05, 15


def cap_per_term(df, cap):
    df = df.sample(frac=1.0, random_state=C.SEED)
    return df.groupby("term_id", sort=False).head(cap)


def main():
    base = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset_v5.parquet")
    anchors = base[base["source"].isin(ANCHOR_SOURCES)].copy()
    print(f"capalar: {len(anchors)} satir "
          f"(eski pseudo atildi: {len(base) - len(anchors)})")

    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["term_id", "item_id"],
                      dtype=str)
    s = np.load(C.ARTIFACTS_DIR / "ce_v6_test.npy")
    assert len(s) == len(sub)

    plp = cap_per_term(sub[s > PL_POS_THR].copy(), PL_POS_PER_TERM)
    plp["label"], plp["source"] = 1.0, "pl_pos"
    pln = cap_per_term(sub[s < PL_NEG_THR].copy(), PL_NEG_PER_TERM)
    pln["label"], pln["source"] = 0.0, "pl_neg"
    print(f"v6 ogretmen kapsami: pos {(s > PL_POS_THR).sum()} -> {len(plp)}  "
          f"neg {(s < PL_NEG_THR).sum()} -> {len(pln)}")

    ds = pd.concat([anchors, plp, pln], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)

    print(f"v7 dataset: {len(ds)} satir  pozitif={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v7.parquet", index=False)
    print("kaydedildi: train_dataset_v7.parquet")


if __name__ == "__main__":
    main()
