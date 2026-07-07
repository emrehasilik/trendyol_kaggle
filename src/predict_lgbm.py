"""Faz 2: LightGBM ile 3.36M test cifti skorla, A1 submission yaz.

Kullanim: python predict_lgbm.py [threshold]
Threshold verilmezse train_lgbm'in proxy sweep'inde bulunan deger elle girilmeli.
Ham skorlar stacker icin artifacts/lgbm_test_scores.npy'ye kaydedilir.
"""
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from features_lean import LeanFeatures, FEATURE_COLS


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    t0 = time.time()
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    lf = LeanFeatures()
    model = lgb.Booster(model_file=str(C.MODELS_DIR / "lgbm_a1.txt"))
    print(f"test={len(sub)}  ({time.time()-t0:.0f}s)")

    X = lf.transform(sub)
    print(f"featurelar cikti ({time.time()-t0:.0f}s)")
    scores = model.predict(X[FEATURE_COLS])
    np.save(C.ARTIFACTS_DIR / "lgbm_test_scores.npy", scores.astype(np.float32))

    pred = (scores > thr).astype(int)
    out = C.OUTPUT_DIR / f"sub_a1_lgbm_thr{thr:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
    print(f"pozitif orani={pred.mean():.3f}  yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
