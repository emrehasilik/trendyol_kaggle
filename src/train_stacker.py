"""Faz 4: stacker — CE skoru + LGBM skoru + term-ici liste istatistikleri.

Term istatistikleri liste-boyu bagimsiz formlarda (persentil rank, z-skor,
max'a oran) — train listeleri ~uzun, test listeleri ~100'luk oldugu icin sart.
GroupKFold(5, term_id). Proxy uzerinde ablation raporu verir.

Cikti: models/stacker.txt, artifacts/stacker_proxy_scores.npz,
       artifacts/stacker_test_scores.npy (test skorlari da uretilir)
"""
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import config as C
import eval_proxy

STACK_COLS = ["ce", "lgbm", "ce_rank_pct", "ce_z", "ce_ratio_max", "term_mean", "term_std"]

LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
    "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1, "seed": C.SEED,
}


def term_stats(df, score_col="ce"):
    g = df.groupby("term_id")[score_col]
    df["ce_rank_pct"] = g.rank(pct=True).astype(np.float32)
    df["term_mean"] = g.transform("mean").astype(np.float32)
    df["term_std"] = g.transform("std").fillna(0).astype(np.float32)
    df["term_max"] = g.transform("max").astype(np.float32)
    df["ce_z"] = ((df[score_col] - df["term_mean"]) /
                  df["term_std"].replace(0, 1)).astype(np.float32)
    df["ce_ratio_max"] = (df[score_col] / df["term_max"].replace(0, 1)).astype(np.float32)
    return df


def main():
    t0 = time.time()
    ce = np.load(C.ARTIFACTS_DIR / "ce_oof.npz", allow_pickle=True)
    lg = np.load(C.ARTIFACTS_DIR / "lgbm_oof.npz", allow_pickle=True)
    assert (ce["term_id"] == lg["term_id"]).all() and (ce["item_id"] == lg["item_id"]).all(), \
        "ce_oof ile lgbm_oof satir sirasi uyusmuyor"

    df = pd.DataFrame({"term_id": ce["term_id"], "ce": ce["oof"].astype(np.float32),
                       "lgbm": lg["oof"].astype(np.float32), "y": ce["y"].astype(np.int8)})
    df = term_stats(df)
    y = df["y"].values
    groups = df["term_id"].values

    # proxy tarafi
    cep = np.load(C.ARTIFACTS_DIR / "ce_proxy_scores.npz", allow_pickle=True)
    lgp = np.load(C.ARTIFACTS_DIR / "lgbm_proxy_scores.npz", allow_pickle=True)
    dfp = pd.DataFrame({"term_id": cep["term_id"], "ce": cep["scores"].astype(np.float32),
                        "lgbm": lgp["scores"].astype(np.float32)})
    dfp = term_stats(dfp)
    yp = cep["y"].astype(np.int8)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))
    proxy_scores = np.zeros(len(dfp))
    best_iters = []
    for fold, (tr, va) in enumerate(gkf.split(df, y, groups)):
        dtr = lgb.Dataset(df.iloc[tr][STACK_COLS], label=y[tr])
        dva = lgb.Dataset(df.iloc[va][STACK_COLS], label=y[va], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(df.iloc[va][STACK_COLS])
        proxy_scores += m.predict(dfp[STACK_COLS]) / 5
        best_iters.append(m.best_iteration)
        print(f"  fold {fold}: best_iter={m.best_iteration} "
              f"f1={f1_score(y[va], (oof[va] > 0.5).astype(int), average='macro'):.4f}")

    eval_proxy.report("stacker OOF", y, oof)
    r = eval_proxy.report("stacker LB-proxy", yp, proxy_scores)
    # kiyas: tek basina CE proxy skoru
    eval_proxy.report("(kiyas) CE-only LB-proxy", yp, dfp["ce"].values)

    final = lgb.train(LGB_PARAMS, lgb.Dataset(df[STACK_COLS], label=y),
                      num_boost_round=int(np.mean(best_iters)))
    final.save_model(str(C.MODELS_DIR / "stacker.txt"))
    np.savez(C.ARTIFACTS_DIR / "stacker_proxy_scores.npz", scores=proxy_scores, y=yp,
             term_id=cep["term_id"])

    # ---- test tarafi
    ce_test = np.load(C.ARTIFACTS_DIR / "ce_test_mean.npy")
    lgbm_test = np.load(C.ARTIFACTS_DIR / "lgbm_test_scores.npy")
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["id", "term_id"], dtype=str)
    dft = pd.DataFrame({"term_id": sub["term_id"], "ce": ce_test.astype(np.float32),
                        "lgbm": lgbm_test.astype(np.float32)})
    dft = term_stats(dft)
    test_scores = final.predict(dft[STACK_COLS])
    np.save(C.ARTIFACTS_DIR / "stacker_test_scores.npy", test_scores.astype(np.float32))

    thr = r["best_thr"]
    pred = (test_scores > thr).astype(int)
    out = C.OUTPUT_DIR / f"sub_a3_stacker_thr{thr:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
    print(f"pozitif orani={pred.mean():.3f}  yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
