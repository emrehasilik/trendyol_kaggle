"""v2 hazirlik: ANN komsularini v1 CE ile puanlayip hard-negatif adayi cikar.

v1'de ANN top-200 negatif havuzundan tamamen yasaklanmisti (false-neg korkusu).
Gercek LB (0.822 vs proxy 0.9219) test negatiflerinin onemli kisminin tam bu
"cok benzer" bantta oldugunu gosterdi. Cozum: yasagi kor kuraldan modele
devret - ANN komsularindan v1 CE'nin dusuk skor verdikleri hard negatif olur.

Cikti: artifacts/ann_vet_scores.npz (term_id, i_row, sim, score)
GPU ~25-30 dk (~700K cift x 2 fold).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

import config as C
from train_ce import TokenCache, score_pairs

DEVICE = "cuda"
RANK_SKIP = 5      # en yakin 5 komsu buyuk olasilikla gercek pozitif, dokunma
PER_TERM = 40      # term basina rastgele aday
VET_FOLDS = (0, 2) # OOF'u en iyi 2 fold'un ortalamasi vetleme icin yeterli

# "python vet_ann_negatives.py v2" -> ce_v2 fold'lariyla vetler,
# ann_vet_scores_v2.npz yazar. Ayni SEED = ayni 638K aday (kiyaslanabilir).
VETTER = next((a for a in sys.argv[1:] if a.startswith("v")), "")
MODEL_PREFIX = f"ce_{VETTER}" if VETTER else "ce"
OUT_NAME = f"ann_vet_scores_{VETTER}.npz" if VETTER else "ann_vet_scores.npz"


def main():
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    cache = TokenCache()

    ann = np.load(C.ARTIFACTS_DIR / "ann_train.npz", allow_pickle=True)
    ann_idx = ann["idx"]
    ann_sim = ann["sim"].astype(np.float32)
    ann_terms = ann["term_ids"]

    proxy = set(np.load(C.ARTIFACTS_DIR / "proxy_terms.npy",
                        allow_pickle=True).tolist())

    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)
    train["i_row"] = train["item_id"].map(cache.i_index)
    pos_map = train.groupby("term_id")["i_row"].apply(set).to_dict()

    ct, ci, csim, ctid = [], [], [], []
    for r, term_id in enumerate(ann_terms):
        if term_id in proxy:
            continue
        pos = pos_map.get(term_id, set())
        cand = ann_idx[r, RANK_SKIP:].astype(np.int64)
        sims = ann_sim[r, RANK_SKIP:]
        keep = np.array([c not in pos for c in cand])
        cand, sims = cand[keep], sims[keep]
        if len(cand) == 0:
            continue
        if len(cand) > PER_TERM:
            pick = rng.choice(len(cand), size=PER_TERM, replace=False)
            cand, sims = cand[pick], sims[pick]
        ct.append(np.full(len(cand), cache.t_index[term_id], dtype=np.int64))
        ci.append(cand)
        csim.append(sims)
        ctid.append(np.full(len(cand), term_id, dtype=object))

    t_rows = np.concatenate(ct)
    i_rows = np.concatenate(ci)
    sims = np.concatenate(csim)
    term_ids = np.concatenate(ctid)
    print(f"aday: {len(t_rows)} cift, {len(ct)} term  ({time.time()-t0:.0f}s)",
          flush=True)

    scores = np.zeros(len(t_rows), dtype=np.float32)
    for fold in VET_FOLDS:
        model = AutoModelForSequenceClassification.from_pretrained(
            C.CE_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
        model.load_state_dict(torch.load(
            C.MODELS_DIR / f"{MODEL_PREFIX}_fold{fold}.pt",
            map_location=DEVICE, weights_only=True))
        scores += score_pairs(model, t_rows, i_rows, cache,
                              desc=f"vet f{fold}") / len(VET_FOLDS)
        del model
        torch.cuda.empty_cache()

    np.savez(C.ARTIFACTS_DIR / OUT_NAME,
             term_id=term_ids, i_row=i_rows, sim=sims, score=scores)
    pcts = [5, 25, 50, 75, 95]
    q = np.percentile(scores, pcts)
    print("ce skor dagilimi: " + " ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, q)))
    print(f"skor<0.30 (hard-neg adayi): {(scores < 0.30).sum()}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
