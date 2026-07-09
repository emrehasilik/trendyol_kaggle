"""Gri bolge sensusu + metadata (cinsiyet/yas) celiski kurallarinin dogrulamasi.

Kurallarin guvenilirligi GERCEK pozitiflerle olculur: bir kural gercek
pozitiflerin anlamli bir kismini negatife cevirecekse guvensizdir.
"""
import re

import numpy as np
import pandas as pd

import config as C

# sorgudan hedef cinsiyet/yas cikarimi (kelime sinirlariyla, cok muhafazakar)
MALE_RE = re.compile(r"\berkek\b")
FEMALE_RE = re.compile(r"\b(kadin|kadın|bayan)\b")
BABY_RE = re.compile(r"\bbebek\b")
CHILD_RE = re.compile(r"\b(cocuk|çocuk|kiz|kız)\b")


def query_signals(q):
    male = bool(MALE_RE.search(q))
    female = bool(FEMALE_RE.search(q))
    if male and female:
        male = female = False  # "erkek kadin ortak" gibi belirsizler disarida
    baby = bool(BABY_RE.search(q))
    return male, female, baby


def main():
    terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
    terms["query"] = terms["query"].str.lower()
    sig = terms["query"].map(query_signals)
    terms["q_male"] = [s[0] for s in sig]
    terms["q_female"] = [s[1] for s in sig]
    terms["q_baby"] = [s[2] for s in sig]
    tmap = terms.set_index("term_id")[["q_male", "q_female", "q_baby"]]

    items = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                        usecols=["item_id", "gender", "age_group"])
    for c in ("gender", "age_group"):
        items[c] = items[c].str.lower()
    imap = items.set_index("item_id")

    def enrich(df):
        df = df.join(tmap, on="term_id").join(imap, on="item_id")
        df["mm_gender"] = ((df["q_male"] & (df["gender"] == "kadın")) |
                           (df["q_female"] & (df["gender"] == "erkek")))
        df["mm_baby"] = df["q_baby"] & (df["age_group"] == "yetişkin")
        return df

    # ---- 1) kural guvenilirligi: gercek pozitifler uzerinde yanlis-alarm orani
    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)
    train = enrich(train)
    n_m = train["q_male"].sum() + train["q_female"].sum()
    print(f"[TRAIN gercek pozitifler] cinsiyetli sorgu cifti: {n_m}")
    print(f"  cinsiyet celiskisi: {train['mm_gender'].sum()} "
          f"(cinsiyetli ciftlerin %{100 * train['mm_gender'].sum() / max(n_m, 1):.2f})")
    n_b = train["q_baby"].sum()
    print(f"  bebek sorgulu cift: {n_b}  yas celiskisi: {train['mm_baby'].sum()} "
          f"(%{100 * train['mm_baby'].sum() / max(n_b, 1):.2f})")

    # ---- 2) test tarafi: gri bolge sensusu + kural kapsami
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str,
                      usecols=["term_id", "item_id"])
    ens = np.load(C.ARTIFACTS_DIR / "ens_v234_test.npy")
    sub = enrich(sub)
    sub["s"] = ens

    thr25 = np.quantile(ens, 0.75)
    print(f"\n[TEST] rate25 esigi={thr25:.4f}")
    for lo, hi in [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80)]:
        m = (ens > lo) & (ens < hi)
        print(f"  gri bolge ({lo},{hi}): {m.sum()} cift (%{100 * m.mean():.1f})")

    for name, col in [("cinsiyet", "mm_gender"), ("bebek-yas", "mm_baby")]:
        mm = sub[sub[col]]
        pos_side = (mm["s"] > thr25).sum()
        print(f"\n  {name} celiskili test cifti: {len(mm)} "
              f"(pozitif tahminli: {pos_side})")
        if len(mm):
            q = np.percentile(mm["s"], [25, 50, 75, 95])
            print(f"    skor dagilimi p25={q[0]:.3f} p50={q[1]:.3f} "
                  f"p75={q[2]:.3f} p95={q[3]:.3f}")


if __name__ == "__main__":
    main()
