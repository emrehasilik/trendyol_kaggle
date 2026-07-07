"""Faz 3: CE fold modelleriyle 3.36M test ciftini skorla (resumable).

Her fold icin artifacts/ce_test_fold{i}.npy yazilir (varsa atlanir).
Sonda 3 foldun ortalamasi + proxy-sweep esigiyle A2 submission uretilir.

Kullanim: python infer_ce.py [threshold]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

import config as C
import eval_proxy
from train_ce import TokenCache, score_pairs

DEVICE = "cuda"


def main():
    thr_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    cache = TokenCache()
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    t_rows = sub["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = sub["item_id"].map(cache.i_index).values.astype(np.int64)
    sub_ids = sub["id"].values
    del sub
    print(f"test={len(t_rows)}  ({time.time()-t0:.0f}s)")

    mean_scores = np.zeros(len(t_rows), dtype=np.float32)
    for fold in range(C.CE_N_FOLDS):
        path = C.ARTIFACTS_DIR / f"ce_test_fold{fold}.npy"
        if path.exists():
            print(f"fold {fold}: test skorlari mevcut, atlaniyor")
            mean_scores += np.load(path) / C.CE_N_FOLDS
            continue
        model = AutoModelForSequenceClassification.from_pretrained(
            C.CE_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
        model.load_state_dict(torch.load(C.MODELS_DIR / f"ce_fold{fold}.pt",
                                         map_location=DEVICE, weights_only=True))
        scores = score_pairs(model, t_rows, i_rows, cache, desc=f"test f{fold}")
        np.save(path, scores)
        mean_scores += scores / C.CE_N_FOLDS
        del model
        torch.cuda.empty_cache()
        print(f"fold {fold} test skorlandi ({time.time()-t0:.0f}s)")

    np.save(C.ARTIFACTS_DIR / "ce_test_mean.npy", mean_scores)

    # esik: arg > proxy sweep
    if thr_arg is None:
        pz = np.load(C.ARTIFACTS_DIR / "ce_proxy_scores.npz")
        thr, f1, _ = eval_proxy.sweep_threshold(pz["y"], pz["scores"])
        print(f"proxy sweep esigi: {thr:.2f} (proxy f1={f1:.4f})")
    else:
        thr = thr_arg

    pred = (mean_scores > thr).astype(int)
    out = C.OUTPUT_DIR / f"sub_a2_ce_thr{thr:.2f}.csv"
    pd.DataFrame({"id": sub_ids, "prediction": pred}).to_csv(out, index=False)
    print(f"pozitif orani={pred.mean():.3f}  yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
