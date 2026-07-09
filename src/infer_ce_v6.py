"""v6: 3.36M test ciftini buyuk govdeyle skorla (chunk-bazli resumable) + submission.

- Cift listesi uzunluga gore SIRALANIR, 262144'luk parcalara bolunur; her parca
  artifacts/ce_{tag}_test_chunk{k:04d}.npy olarak Drive'a yazilir (varsa atlanir)
  -> oturum kopsa bile en fazla 1 parca kaybedilir.
- Esik SABIT KURAL geregi %25 pozitif oran: thr = quantile(skorlar, 0.75)
  (HANDOFF DERS #4; oran LB'de kalibre edildi).

Kullanim: python infer_ce_v6.py [--tag v6] [--rate 0.25]
Cikti   : artifacts/ce_{tag}_test.npy, output/sub_{tag}_rate{25}.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

import config as C
from train_ce_v6 import TokenCacheV6, amp_setup, score_pairs_v6

DEVICE = "cuda"
CHUNK = 262_144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--rate", type=float, default=0.25)
    args = ap.parse_args()

    t0 = time.time()
    cache = TokenCacheV6()
    amp_dtype, _ = amp_setup()

    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    t_rows = sub["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = sub["item_id"].map(cache.i_index).values.astype(np.int64)
    sub_ids = sub["id"].values
    del sub
    n = len(t_rows)
    print(f"test={n}  ({time.time()-t0:.0f}s)")

    # global uzunluk sirasi: her chunk homojen uzunlukta -> verimli padding
    order = np.argsort(cache.pair_len(t_rows, i_rows), kind="stable")
    n_chunks = (n + CHUNK - 1) // CHUNK

    model = None
    sorted_scores = np.empty(n, dtype=np.float32)
    for k in range(n_chunks):
        path = C.ARTIFACTS_DIR / f"ce_{args.tag}_test_chunk{k:04d}.npy"
        sl = order[k * CHUNK:(k + 1) * CHUNK]
        if path.exists():
            sorted_scores[k * CHUNK:k * CHUNK + len(sl)] = np.load(path)
            print(f"chunk {k}/{n_chunks} mevcut, atlaniyor")
            continue
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                C.CE_V6_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
            model.load_state_dict(torch.load(
                C.MODELS_DIR / f"ce_{args.tag}_final.pt",
                map_location=DEVICE, weights_only=True))
        s = score_pairs_v6(model, t_rows[sl], i_rows[sl], cache, amp_dtype,
                           desc=f"chunk {k}")
        np.save(path, s.astype(np.float16))
        sorted_scores[k * CHUNK:k * CHUNK + len(sl)] = s
        print(f"chunk {k+1}/{n_chunks} skorlandi ({time.time()-t0:.0f}s)", flush=True)

    scores = np.empty(n, dtype=np.float32)
    scores[order] = sorted_scores
    np.save(C.ARTIFACTS_DIR / f"ce_{args.tag}_test.npy", scores)

    thr = float(np.quantile(scores, 1.0 - args.rate))
    pred = (scores > thr).astype(int)
    gray = float(((scores > 0.05) & (scores < 0.95)).mean())
    out = C.OUTPUT_DIR / f"sub_{args.tag}_rate{int(args.rate*100)}.csv"
    pd.DataFrame({"id": sub_ids, "prediction": pred}).to_csv(out, index=False)
    print(f"esik={thr:.4f}  pozitif_orani={pred.mean():.4f}  "
          f"gri_bolge(0.05-0.95)={gray:.3f}")
    print(f"yazildi: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
