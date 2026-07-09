"""Liste-ici kategori oylamasi: emin pozitiflerin kategorileri gri bolgeyi yargilar.

Hipotez: bir sorgunun emin pozitifleri (ens>0.9) hangi yaprak kategorilerdeyse,
ayni sorgunun gri (0.05-0.95) ciftlerinden o kategorilerde olanlar buyuk
olasilikla pozitif, olmayanlar negatiftir.

Dogrulama proxy'de GERCEK pozitif etiketlerle yapilir:
  P(destek | gri, y=1)  vs  P(destek | gri, y=0)  ayrimi guclu mu?
Ayrica destek bonusunun proxy macro-F1'e etkisi taranir.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as C

CONF = 0.90          # emin pozitif esigi
GRAY_LO, GRAY_HI = 0.05, 0.95


def leaf_of(cat):
    return cat.rsplit("/", 1)[-1].strip() if cat else ""


def add_support(df, items_leaf):
    """df: term_id, item_id, s. Liste-ici kategori destegi ekler."""
    df = df.copy()
    df["leaf"] = df["item_id"].map(items_leaf)
    conf = df[df["s"] > CONF]
    conf_cats = conf.groupby("term_id")["leaf"].agg(set).to_dict()
    df["support"] = [
        leaf in conf_cats.get(t, set())
        for t, leaf in zip(df["term_id"].values, df["leaf"].values)
    ]
    df["has_conf"] = df["term_id"].map(lambda t: t in conf_cats)
    return df


def main():
    items = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                        usecols=["item_id", "category"])
    items_leaf = pd.Series(items["category"].map(leaf_of).values,
                           index=items["item_id"]).to_dict()

    # ---------- proxy dogrulamasi (gercek pozitif etiketler)
    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    zs = [np.load(C.ARTIFACTS_DIR / f)["scores"]
          for f in ("ce_v2_proxy_scores.npz", "ce_v3_proxy_scores.npz",
                    "ce_v4_proxy_scores.npz")]
    proxy["s"] = np.mean(zs, axis=0)
    proxy = add_support(proxy[["term_id", "item_id", "label", "s"]], items_leaf)

    gray = proxy[(proxy["s"] > GRAY_LO) & (proxy["s"] < GRAY_HI) & proxy["has_conf"]]
    g1 = gray[gray["label"] == 1]
    g0 = gray[gray["label"] == 0]
    print(f"[PROXY gri bolge] n={len(gray)}  gercek-poz={len(g1)}  sentetik-neg={len(g0)}")
    print(f"  P(destek | y=1) = {g1['support'].mean():.3f}")
    print(f"  P(destek | y=0) = {g0['support'].mean():.3f}")

    # bonus taramasi: gri bolgede destege gore skoru kaydir
    y = proxy["label"].values.astype(int)
    base = proxy["s"].values
    gmask = ((base > GRAY_LO) & (base < GRAY_HI) & proxy["has_conf"].values)
    sup = proxy["support"].values
    best0 = max(f1_score(y, (base > t).astype(int), average="macro")
                for t in np.arange(0.05, 0.96, 0.01))
    print(f"  proxy taban macro-F1 (en iyi esik): {best0:.4f}")
    for bonus in (0.1, 0.2, 0.3, 0.4):
        for malus in (0.0, -0.1, -0.2, -0.3):
            adj = base.copy()
            adj[gmask & sup] += bonus
            adj[gmask & ~sup] += malus
            f = max(f1_score(y, (adj > t).astype(int), average="macro")
                    for t in np.arange(0.05, 0.96, 0.01))
            if f > best0 + 0.0005:
                print(f"  bonus={bonus:+.1f} malus={malus:+.1f}: {f:.4f}  (+{f-best0:.4f})")

    # ---------- test tarafi kapsam
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str,
                      usecols=["term_id", "item_id"])
    sub["s"] = np.load(C.ARTIFACTS_DIR / "ens_v234_test.npy")
    sub = add_support(sub, items_leaf)
    gmask_t = ((sub["s"] > GRAY_LO) & (sub["s"] < GRAY_HI) & sub["has_conf"]).values
    print(f"\n[TEST] gri+conf'lu cift: {gmask_t.sum()} "
          f"(destekli: {(gmask_t & sub['support'].values).sum()})")
    sub[["support"]].to_parquet(C.ARTIFACTS_DIR / "cat_support_test.parquet")
    np.save(C.ARTIFACTS_DIR / "cat_gray_mask_test.npy", gmask_t)
    print("kaydedildi: cat_support_test.parquet, cat_gray_mask_test.npy")


if __name__ == "__main__":
    main()
