"""Typo-duzeltmeli yeniden skorlama + gercek etiketli dogrulama.

Degisen sorgular (terms_corrected.csv, changed=True) v3 fold'lariyla yeniden
skorlanir. Nihai skor = max(orijinal ensemble, duzeltilmis v3 ortalamasi):
max-blend yalnizca yukari cekebilir -> typo kaynakli kacan pozitifleri
kurtarir; yanlis duzeltmenin zarari sinirli kalir.

Once proxy (gercek pozitif etiketli) uzerinde olculur, sonra test dosyasi
uretilir: output/sub_ens_v234_typofix_rate25.csv
"""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as C
from train_ce import TokenCache, score_pairs

DEVICE = "cuda"


def build_corrected_cache(base: TokenCache, corrected_terms):
    """Degisen sorgularin duzeltilmis metnini tokenize edip base cache'in
    term tarafini degistiren shim cache dondurur."""
    tok = AutoTokenizer.from_pretrained(C.CE_MODEL_NAME, cache_dir=C.HF_CACHE)
    enc = tok(list(corrected_terms["corrected"].values), add_special_tokens=False,
              truncation=True, max_length=32)["input_ids"]
    vals = np.asarray([i for ids in enc for i in ids], dtype=np.uint32)
    off = np.cumsum([0] + [len(ids) for ids in enc]).astype(np.int64)
    t_index = {tid: k for k, tid in enumerate(corrected_terms["term_id"].values)}
    return SimpleNamespace(t_vals=vals, t_off=off,
                           i_vals=base.i_vals, i_off=base.i_off,
                           cls_id=base.cls_id, sep_id=base.sep_id,
                           pad_id=base.pad_id), t_index


def rescore(pairs, cache_shim, t_index, base_cache, desc):
    t_rows = pairs["term_id"].map(t_index).values.astype(np.int64)
    i_rows = pairs["item_id"].map(base_cache.i_index).values.astype(np.int64)
    out = np.zeros(len(pairs), dtype=np.float32)
    for fold in range(C.CE_N_FOLDS):
        model = AutoModelForSequenceClassification.from_pretrained(
            C.CE_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
        model.load_state_dict(torch.load(
            C.MODELS_DIR / f"ce_v3_fold{fold}.pt",
            map_location=DEVICE, weights_only=True))
        out += score_pairs(model, t_rows, i_rows, cache_shim,
                           desc=f"{desc} f{fold}") / C.CE_N_FOLDS
        del model
        torch.cuda.empty_cache()
    return out


def sweep(y, s):
    return max(f1_score(y, (s > t).astype(int), average="macro")
               for t in np.arange(0.05, 0.96, 0.01))


def main():
    t0 = time.time()
    base = TokenCache()
    tc = pd.read_csv(C.ARTIFACTS_DIR / "terms_corrected.csv", dtype={"term_id": str})
    changed = tc[tc["changed"] == True][["term_id", "corrected"]]  # noqa: E712
    shim, t_index = build_corrected_cache(base, changed)
    changed_set = set(changed["term_id"].values)
    print(f"degisen sorgu: {len(changed)}  ({time.time()-t0:.0f}s)")

    # ---------- proxy dogrulamasi
    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    zs = [np.load(C.ARTIFACTS_DIR / f)["scores"]
          for f in ("ce_v2_proxy_scores.npz", "ce_v3_proxy_scores.npz",
                    "ce_v4_proxy_scores.npz")]
    proxy["s"] = np.mean(zs, axis=0)
    pmask = proxy["term_id"].isin(changed_set).values
    print(f"proxy'de degisen term cifti: {pmask.sum()} "
          f"({proxy[pmask]['term_id'].nunique()} term)")
    if pmask.sum():
        new_s = rescore(proxy[pmask], shim, t_index, base, "proxy-typofix")
        adj = proxy["s"].values.copy()
        adj[pmask] = np.maximum(adj[pmask], new_s)
        y = proxy["label"].values.astype(int)
        print(f"[PROXY tum]      once={sweep(y, proxy['s'].values):.4f}  "
              f"sonra={sweep(y, adj):.4f}")
        ym, sm, am = y[pmask], proxy["s"].values[pmask], adj[pmask]
        print(f"[PROXY etkilenen] once={sweep(ym, sm):.4f}  sonra={sweep(ym, am):.4f}")
        pos = ym == 1
        print(f"  etkilenen gercek pozitif ort skor: {sm[pos].mean():.3f} -> "
              f"{am[pos].mean():.3f}  |  sentetik neg: {sm[~pos].mean():.3f} -> "
              f"{am[~pos].mean():.3f}")

    # ---------- test tarafi
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    ens = np.load(C.ARTIFACTS_DIR / "ens_v234_test.npy")
    tmask = sub["term_id"].isin(changed_set).values
    print(f"\ntestte degisen term cifti: {tmask.sum()} "
          f"({sub[tmask]['term_id'].nunique()} term)  ({time.time()-t0:.0f}s)")
    new_s = rescore(sub[tmask], shim, t_index, base, "test-typofix")
    adj = ens.copy()
    adj[tmask] = np.maximum(adj[tmask], new_s)
    np.save(C.ARTIFACTS_DIR / "ens_v234_typofix_test.npy", adj)

    thr = np.quantile(adj, 0.75)
    pred = (adj > thr).astype(int)
    out = C.OUTPUT_DIR / "sub_ens_v234_typofix_rate25.csv"
    pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
    flips = int((pred != (ens > np.quantile(ens, 0.75)).astype(int)).sum())
    print(f"esik={thr:.4f}  pozitif={pred.mean():.3f}  degisen tahmin={flips}")
    print(f"yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
