"""v9 dataseti: v5 tarifine LLM-hakem etiketli gri bolge ciftlerini ekle.

v8 kaniti (HANDOFF bolum 8): LLM hakem gri bolgede CE'den iyi
(LB 0.855 -> 0.860 alpha0.7 -> 0.861 alpha1.0). v9 bu bilgiyi MODELE tasir:
LLM etiketli ciftler egitim setine REP kez tekrarla eklenir (sinir bolgesi
agirligi); model duzeltmeleri yargilanmamis ciftlere de genellestirir.
Taban veri AYNEN train_dataset_v5 (0.855'i ureten tarif) -> tek degisken deney.

- LLM etiketi SOFT kullanilir (p aynen label olur); kararsiz bolge
  (P_NEG < p < P_POS) atilir.
- v7'nin basarisiz v6-pseudo tarifi KULLANILMAZ (ayni soy, bilgi yok).

Girdi : artifacts/train_dataset_v5.parquet, artifacts/llm_{name}_idx.npy + chunk'lar
Cikti : artifacts/train_dataset_v9.parquet
Kullanim: python build_dataset_v9.py [--names v6,v6r2] [--rep 3]
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from llm_judge_v8 import load_judgments

P_POS, P_NEG = 0.7, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="v6")
    ap.add_argument("--rep", type=int, default=3)
    args = ap.parse_args()

    base = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset_v5.parquet")
    print(f"taban (v5): {len(base)} satir  poz={base['label'].mean():.3f}")

    names = [nm for nm in args.names.split(",") if nm]
    pairs = [load_judgments(nm) for nm in names]
    ji = np.concatenate([p[0] for p in pairs])
    jp = np.concatenate([p[1] for p in pairs])
    assert len(np.unique(ji)) == len(ji), "turlar arasi cift tekrari var"

    keep = (jp >= P_POS) | (jp <= P_NEG)
    print(f"LLM etiketleri ({'+'.join(names)}): {len(ji)} yargi, "
          f"{int(keep.sum())} kesin (%{100 * (1 - keep.mean()):.1f} kararsiz atildi)  "
          f"poz orani={float((jp[keep] >= P_POS).mean()):.3f}")

    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["term_id", "item_id"],
                      dtype=str)
    llm = sub.iloc[ji[keep]].copy()
    llm["label"] = jp[keep].astype(np.float64)  # soft etiket
    llm["source"] = "llm"
    llm = pd.concat([llm] * args.rep, ignore_index=True)
    print(f"llm satirlari: {len(llm)} (rep={args.rep})")

    ds = pd.concat([base, llm], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    print(f"v9 dataset: {len(ds)} satir  ort etiket={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v9.parquet", index=False)
    print("kaydedildi: train_dataset_v9.parquet")


if __name__ == "__main__":
    main()
