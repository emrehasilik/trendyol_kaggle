"""Faz 2: 8-feature LightGBM'i yeni (dagilim-esli) veri setiyle egit.

Eski 398 MB FeatureBuilder pickle'i RAM'e sigmadigi icin yalin pipeline
(features_lean.LeanFeatures) kullanilir.
Cikti: models/lgbm_a1.txt, artifacts/lgbm_oof.npz, artifacts/lgbm_proxy_scores.npz
"""
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import config as C
from features_lean import LeanFeatures, FEATURE_COLS
import eval_proxy

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": C.SEED,
}


def main():
    t0 = time.time()
    ds = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset.parquet")
    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    lf = LeanFeatures()
    print(f"veri: train={len(ds)} proxy={len(proxy)}  ({time.time()-t0:.0f}s)")

    X = lf.transform(ds)
    y = ds["label"].values
    groups = ds["term_id"].values
    Xp = lf.transform(proxy)
    yp = proxy["label"].values
    print(f"featurelar cikti: X={X.shape} Xp={Xp.shape}  ({time.time()-t0:.0f}s)")

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X))
    proxy_scores = np.zeros(len(Xp))
    best_iters = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr][FEATURE_COLS], label=y[tr])
        dva = lgb.Dataset(X.iloc[va][FEATURE_COLS], label=y[va], reference=dtr)
        model = lgb.train(LGB_PARAMS, dtr, num_boost_round=500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = model.predict(X.iloc[va][FEATURE_COLS])
        proxy_scores += model.predict(Xp[FEATURE_COLS]) / 5
        best_iters.append(model.best_iteration)
        f1 = f1_score(y[va], (oof[va] > 0.5).astype(int), average="macro")
        print(f"  fold {fold}: best_iter={model.best_iteration} macro_f1={f1:.4f}")

    eval_proxy.report("OOF (ornekli negatifler)", y, oof)
    eval_proxy.report("LB-proxy (100'luk listeler)", yp, proxy_scores)

    final = lgb.train(LGB_PARAMS, lgb.Dataset(X[FEATURE_COLS], label=y),
                      num_boost_round=int(np.mean(best_iters)))
    final.save_model(str(C.MODELS_DIR / "lgbm_a1.txt"))
    np.savez(C.ARTIFACTS_DIR / "lgbm_oof.npz", oof=oof, y=y,
             term_id=ds["term_id"].values, item_id=ds["item_id"].values)
    np.savez(C.ARTIFACTS_DIR / "lgbm_proxy_scores.npz", scores=proxy_scores, y=yp,
             term_id=proxy["term_id"].values, item_id=proxy["item_id"].values)
    print(f"kaydedildi. toplam {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
