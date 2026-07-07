"""v3 dataseti: pseudo-label dongusunun 2. turu, bu kez v2 skorlariyla.

v2 LB'de 0.845 aldi (v1-blend: 0.822) -> pseudo-etiketleri ve ANN vetlemesini
daha iyi yargicla (v2) yenileyip ayni receteyle yeniden egitiyoruz:
  - pl_pos / pl_neg : ce_v2_test_mean.npy'den (>0.95 / <0.05), kapsam v2'de
    daha genis cunku v2 testte cok daha kararli (%92.8 emin bolgede).
  - annvet          : ann_vet_scores_v2.npz'den (v2 CE < 0.25, en zor 16/term).
  - pos             : gercek pozitifler (proxy disi, degismez).
  - v1neg           : kucultulmus v1 negatif cekirdegi (dagilim hafizasi).

proxy_terms yine tamamen haric -> proxy kiyasi v1/v2 ile adil kalir.
Cikti: artifacts/train_dataset_v3.parquet (term_id, item_id, label, source)
"""
import numpy as np
import pandas as pd

import config as C

PL_POS_THR = 0.95
PL_NEG_THR = 0.05
PL_POS_PER_TERM = 18
PL_NEG_PER_TERM = 15
ANN_VET_THR = 0.25
ANN_PER_TERM = 16
V1_NEG_SAMPLE = 120_000


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
    ct = np.load(C.ARTIFACTS_DIR / "ce_v2_test_mean.npy")
    assert len(ct) == len(sub), "v2 test skorlari ile submission_pairs uyusmuyor"

    plp = cap_per_term(sub[ct > PL_POS_THR].copy(), PL_POS_PER_TERM)
    plp["label"], plp["source"] = 1.0, "pl_pos"
    pln = cap_per_term(sub[ct < PL_NEG_THR].copy(), PL_NEG_PER_TERM)
    pln["label"], pln["source"] = 0.0, "pl_neg"

    ds = pd.concat([pos, annvet, v1neg, plp, pln], ignore_index=True)
    assert not ds["term_id"].isin(proxy).any(), "proxy termi sizmis!"
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)

    print(f"v3 dataset: {len(ds)} satir  pozitif={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v3.parquet", index=False)
    print("kaydedildi: train_dataset_v3.parquet")


if __name__ == "__main__":
    main()
