"""v2 dataseti: gercek pozitifler + CE-onayli ANN hard negatifler + v1 negatif
cekirdegi + test pseudo-labellari.

LB teshisi (0.822 vs proxy 0.9219): sentetik negatifler gercek test
negatiflerinden farkliydi. Iki duzeltme:
  1. pseudo-label: blend'in cok emin oldugu test ciftleri (>0.95 / <0.05)
     egitime girer - model ilk kez GERCEK test dagiliminda antrenman yapar
     (train ve test termleri hic kesismedigi icin bu 32K yeni term demek).
  2. annvet: v1'de yasakli ANN bandindan, v1 CE'nin <0.30 verdigi
     cok-benzer-ama-alakasiz ciftler (vet_ann_negatives.py ciktisi).

proxy_terms tamamen haric -> proxy karsilastirmasi v1 ile birebir adil kalir.
Cikti: artifacts/train_dataset_v2.parquet (term_id, item_id, label, source)
"""
import numpy as np
import pandas as pd

import config as C

PL_POS_THR = 0.95
PL_NEG_THR = 0.05
PL_POS_PER_TERM = 15
PL_NEG_PER_TERM = 13
ANN_VET_THR = 0.30
ANN_PER_TERM = 16
V1_NEG_SAMPLE = 180_000


def cap_per_term(df, cap, sort_col=None):
    """Term basina en fazla `cap` satir; sort_col verilirse en yuksekler."""
    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=False)
    else:
        df = df.sample(frac=1.0, random_state=C.SEED)
    return df.groupby("term_id", sort=False).head(cap)


def main():
    items = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
    proxy = set(np.load(C.ARTIFACTS_DIR / "proxy_terms.npy",
                        allow_pickle=True).tolist())

    # 1) gercek pozitifler (proxy disi)
    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)[["term_id", "item_id"]]
    pos = train[~train["term_id"].isin(proxy)].copy()
    pos["label"], pos["source"] = 1.0, "pos"

    # 2) CE-onayli ANN hard negatifler (en yuksek sim = en zor olanlar oncelikli)
    vet = np.load(C.ARTIFACTS_DIR / "ann_vet_scores.npz", allow_pickle=True)
    vdf = pd.DataFrame({"term_id": vet["term_id"], "i_row": vet["i_row"],
                        "sim": vet["sim"], "score": vet["score"]})
    vdf = cap_per_term(vdf[vdf["score"] < ANN_VET_THR], ANN_PER_TERM, sort_col="sim")
    annvet = pd.DataFrame({"term_id": vdf["term_id"].values,
                           "item_id": items[vdf["i_row"].values.astype(np.int64)],
                           "label": 0.0, "source": "annvet"})

    # 3) v1 negatif cekirdegi (kalibre dagilim hafizasi)
    v1 = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset.parquet")
    v1n = v1[v1["label"] == 0]
    v1n = v1n.sample(n=min(V1_NEG_SAMPLE, len(v1n)), random_state=C.SEED)
    v1neg = pd.DataFrame({"term_id": v1n["term_id"].values,
                          "item_id": v1n["item_id"].values,
                          "label": 0.0, "source": "v1neg"})

    # 4) test pseudo-labellari (blend skorlarindan; satir sirasi birebir ayni)
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["term_id", "item_id"],
                      dtype=str)
    bt = np.load(C.ARTIFACTS_DIR / "blend_test_scores.npy")
    assert len(bt) == len(sub), "blend skorlari ile submission_pairs uyusmuyor"

    plp = cap_per_term(sub[bt > PL_POS_THR].copy(), PL_POS_PER_TERM)
    plp["label"], plp["source"] = 1.0, "pl_pos"
    pln = cap_per_term(sub[bt < PL_NEG_THR].copy(), PL_NEG_PER_TERM)
    pln["label"], pln["source"] = 0.0, "pl_neg"

    ds = pd.concat([pos, annvet, v1neg, plp, pln], ignore_index=True)
    assert not ds["term_id"].isin(proxy).any(), "proxy termi sizmis!"
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)

    print(f"v2 dataset: {len(ds)} satir  pozitif={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v2.parquet", index=False)
    print("kaydedildi: train_dataset_v2.parquet")


if __name__ == "__main__":
    main()
