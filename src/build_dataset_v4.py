"""v4 dataseti: UZLASMA-filtreli pseudo-labellar (v2 ve v3 ortak karari).

v3 LB'de 0.843 aldi (v2: 0.845) -> ayni-model self-training doydu; tek modelin
emin hatalari pseudo-etiketlere sizip pekisiyor. Yeni kural: bir test cifti
ancak IKI model de hemfikirse pseudo-etiket olur:
  pl_pos: mean(v2,v3) > 0.95 VE min(v2,v3) > 0.90
  pl_neg: mean(v2,v3) < 0.05 VE max(v2,v3) < 0.10
Boylece v4 (mDeBERTa, farkli mimari) en temiz mevcut etiketlerle baslar.

Cikti: artifacts/train_dataset_v4.parquet (term_id, item_id, label, source)
"""
import numpy as np
import pandas as pd

import config as C

PL_POS_PER_TERM = 18
PL_NEG_PER_TERM = 15
ANN_VET_THR = 0.25
ANN_PER_TERM = 16
V1_NEG_SAMPLE = 100_000


def cap_per_term(df, cap, sort_col=None):
    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=False)
    else:
        df = df.sample(frac=1.0, random_state=C.SEED)
    return df.groupby("term_id", sort=False).head(cap)


def main():
    items = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
    proxy = set(np.load(C.ARTIFACTS_DIR / "proxy_terms.npy",
                        allow_pickle=True).tolist())

    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)[["term_id", "item_id"]]
    pos = train[~train["term_id"].isin(proxy)].copy()
    pos["label"], pos["source"] = 1.0, "pos"

    vet = np.load(C.ARTIFACTS_DIR / "ann_vet_scores_v2.npz", allow_pickle=True)
    vdf = pd.DataFrame({"term_id": vet["term_id"], "i_row": vet["i_row"],
                        "sim": vet["sim"], "score": vet["score"]})
    vdf = cap_per_term(vdf[vdf["score"] < ANN_VET_THR], ANN_PER_TERM, sort_col="sim")
    annvet = pd.DataFrame({"term_id": vdf["term_id"].values,
                           "item_id": items[vdf["i_row"].values.astype(np.int64)],
                           "label": 0.0, "source": "annvet"})

    v1 = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset.parquet")
    v1n = v1[v1["label"] == 0]
    v1n = v1n.sample(n=min(V1_NEG_SAMPLE, len(v1n)), random_state=C.SEED)
    v1neg = pd.DataFrame({"term_id": v1n["term_id"].values,
                          "item_id": v1n["item_id"].values,
                          "label": 0.0, "source": "v1neg"})

    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["term_id", "item_id"],
                      dtype=str)
    s2 = np.load(C.ARTIFACTS_DIR / "ce_v2_test_mean.npy")
    s3 = np.load(C.ARTIFACTS_DIR / "ce_v3_test_mean.npy")
    assert len(s2) == len(s3) == len(sub)
    mean = (s2 + s3) / 2
    lo = np.minimum(s2, s3)
    hi = np.maximum(s2, s3)

    pos_mask = (mean > 0.95) & (lo > 0.90)
    neg_mask = (mean < 0.05) & (hi < 0.10)
    print(f"uzlasma kapsami: pos {pos_mask.sum()}  neg {neg_mask.sum()}  "
          f"(anlasmazlik |v2-v3|>0.5: {((hi-lo)>0.5).mean()*100:.1f}%)")

    plp = cap_per_term(sub[pos_mask].copy(), PL_POS_PER_TERM)
    plp["label"], plp["source"] = 1.0, "pl_pos"
    pln = cap_per_term(sub[neg_mask].copy(), PL_NEG_PER_TERM)
    pln["label"], pln["source"] = 0.0, "pl_neg"

    ds = pd.concat([pos, annvet, v1neg, plp, pln], ignore_index=True)
    assert not ds["term_id"].isin(proxy).any(), "proxy termi sizmis!"
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)

    print(f"v4 dataset: {len(ds)} satir  pozitif={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v4.parquet", index=False)
    print("kaydedildi: train_dataset_v4.parquet")


if __name__ == "__main__":
    main()
