"""Faz 7: CE v4 (mDeBERTa, tek model) ile 3.36M test ciftini skorla.

Cikti: artifacts/ce_v4_test.npy + sub_v4_ce_thr{thr}.csv
Ensemble submission'lari ayri uretilir (v3+v4 ortalamasi, eval adiminda).

Kullanim: python infer_ce_v4.py [threshold]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

import config as C
import eval_proxy
from train_ce_v4 import MODEL_NAME, TokenCacheM, score_pairs_v4 as score_pairs

DEVICE = "cuda"


def main():
    thr_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    cache = TokenCacheM()
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    t_rows = sub["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = sub["item_id"].map(cache.i_index).values.astype(np.int64)
    sub_ids = sub["id"].values
    del sub
    print(f"test={len(t_rows)}  ({time.time()-t0:.0f}s)")

    path = C.ARTIFACTS_DIR / "ce_v4_test.npy"
    if path.exists():
        print("test skorlari mevcut, atlaniyor")
        scores = np.load(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
        model.load_state_dict(torch.load(C.MODELS_DIR / "ce_v4_full.pt",
                                         map_location=DEVICE, weights_only=True))
        scores = score_pairs(model, t_rows, i_rows, cache, desc="test f0")
        np.save(path, scores)
        print(f"fold 0 test skorlandi ({time.time()-t0:.0f}s)")

    if thr_arg is None:
        pz = np.load(C.ARTIFACTS_DIR / "ce_v4_proxy_scores.npz")
        thr, f1, _ = eval_proxy.sweep_threshold(pz["y"], pz["scores"])
        print(f"proxy sweep esigi: {thr:.2f} (proxy f1={f1:.4f})")
    else:
        thr = thr_arg

    pred = (scores > thr).astype(int)
    out = C.OUTPUT_DIR / f"sub_v4_ce_thr{thr:.2f}.csv"
    pd.DataFrame({"id": sub_ids, "prediction": pred}).to_csv(out, index=False)
    print(f"pozitif orani={pred.mean():.3f}  yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
