"""v5 (konsolidasyon) dataseti: v4 uzlasma-verisi + proxy'nin GERCEK pozitifleri.

Proxy seti pusulaliktan emekli oldu (LB ile art arda celisti) -> sakladigimiz
2000 terimin ~29K gercek pozitifi artik egitime giriyor (+%13 gercek etiket).
Denge icin ayni terimlerin ensemble'in cok emin oldugu (<0.05) liste uyeleri
pseudo-negatif olarak eklenir (yalniz-pozitif terim grubu olusmasin diye).

Cikti: artifacts/train_dataset_v5.parquet
"""
import numpy as np
import pandas as pd

import config as C

PROXY_NEG_PER_TERM = 13
PROXY_NEG_THR = 0.05


def main():
    base = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset_v4.parquet")

    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    zs = [np.load(C.ARTIFACTS_DIR / f)["scores"]
          for f in ("ce_v2_proxy_scores.npz", "ce_v3_proxy_scores.npz",
                    "ce_v4_proxy_scores.npz")]
    proxy["s"] = np.mean(zs, axis=0)

    ppos = proxy[proxy["label"] == 1][["term_id", "item_id"]].copy()
    ppos["label"], ppos["source"] = 1.0, "proxy_pos"

    pneg = proxy[(proxy["label"] == 0) & (proxy["s"] < PROXY_NEG_THR)].copy()
    pneg = pneg.sample(frac=1.0, random_state=C.SEED) \
               .groupby("term_id", sort=False).head(PROXY_NEG_PER_TERM)
    pneg = pneg[["term_id", "item_id"]]
    pneg["label"], pneg["source"] = 0.0, "proxy_neg"

    ds = pd.concat([base, ppos, pneg], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)

    print(f"v5 dataset: {len(ds)} satir  pozitif={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v5.parquet", index=False)
    print("kaydedildi: train_dataset_v5.parquet")


if __name__ == "__main__":
    main()
