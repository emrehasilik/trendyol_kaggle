"""v10 dataseti: LLM denetimli anchor'lar + TUM LLM test turlari + celiski temizligi.

v9 tarifinin (LB 0.877/0.880) uc iyilestirmesi:
1. ANCHOR DENETIMI: annvet/v1neg sentetik negatifleri llm_audit_train.py'den
   gecti; p>=0.7 (false-negatif) satirlar SOFT POZITIFE cevrilir
   (source '{src}_fix'), 0.3<p<0.7 kararsizlar ATILIR, p<=0.3 aynen kalir.
2. CELISKI TEMIZLIGI: taban pl_pos/pl_neg (v4-uzlasma) satirlarindan, LLM'in
   yargiladigi ciftlerle cakisanlar atilir — ayni cifte iki farkli etiket
   gradyani sulandirir; LLM etiketi kazanir.
3. TUM TURLAR: test LLM etiketleri v6,v6r2,v9,v9r2'den (soft, kesin olanlar,
   REP kez tekrar).

Girdi : artifacts/train_dataset_v5.parquet, llm_{name}_* (test), llm_tr_{audit}_*
Cikti : artifacts/train_dataset_v10.parquet
Kullanim: python build_dataset_v10.py [--names v6,v6r2,v9,v9r2] [--audit tr1] [--rep 3]
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from llm_audit_train import load_train_audit
from llm_judge_v8 import load_judgments

P_POS, P_NEG = 0.7, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="v6,v6r2,v9,v9r2")
    ap.add_argument("--audit", default="tr1")
    ap.add_argument("--rep", type=int, default=3)
    args = ap.parse_args()

    base = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset_v5.parquet")
    print(f"taban (v5): {len(base)} satir  poz={base['label'].mean():.3f}")

    # 1) anchor denetimi (satir indeksleri v5 okuma sirasina gore)
    aidx, ap_ = load_train_audit(args.audit)
    flip = aidx[ap_ >= P_POS]
    drop_unc = aidx[(ap_ > P_NEG) & (ap_ < P_POS)]
    base.loc[flip, "label"] = ap_[ap_ >= P_POS].astype(np.float64)
    base.loc[flip, "source"] = base.loc[flip, "source"] + "_fix"
    base = base.drop(index=drop_unc)
    print(f"anchor denetimi ({args.audit}): {len(aidx)} bakildi, "
          f"{len(flip)} false-neg pozitife cevrildi (%{100 * len(flip) / len(aidx):.1f}), "
          f"{len(drop_unc)} kararsiz atildi")

    # 2) test LLM etiketleri (tum turlar)
    names = [nm for nm in args.names.split(",") if nm]
    pairs = [load_judgments(nm) for nm in names]
    ji = np.concatenate([p[0] for p in pairs])
    jp = np.concatenate([p[1] for p in pairs])
    assert len(np.unique(ji)) == len(ji), "turlar arasi cift tekrari var"
    keep = (jp >= P_POS) | (jp <= P_NEG)
    print(f"LLM test etiketleri ({'+'.join(names)}): {len(ji)} yargi, "
          f"{int(keep.sum())} kesin  poz={float((jp[keep] >= P_POS).mean()):.3f}")

    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["term_id", "item_id"],
                      dtype=str)
    llm = sub.iloc[ji[keep]].copy()
    llm["label"] = jp[keep].astype(np.float64)
    llm["source"] = "llm"

    # 3) celiski temizligi: LLM'in yargiladigi ciftlerdeki eski pl_* satirlari
    llm_keys = set(llm["term_id"] + "|" + llm["item_id"])
    pl = base["source"].isin(["pl_pos", "pl_neg"])
    conflict = pl & (base["term_id"] + "|" + base["item_id"]).isin(llm_keys)
    base = base[~conflict]
    print(f"celiski temizligi: {int(conflict.sum())} eski pl satiri atildi")

    llm = pd.concat([llm] * args.rep, ignore_index=True)
    ds = pd.concat([base, llm], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    print(f"v10 dataset: {len(ds)} satir  ort etiket={ds['label'].mean():.3f}  "
          f"term={ds['term_id'].nunique()}")
    print(ds["source"].value_counts().to_string())
    ds.to_parquet(C.ARTIFACTS_DIR / "train_dataset_v10.parquet", index=False)
    print("kaydedildi: train_dataset_v10.parquet")


if __name__ == "__main__":
    main()
