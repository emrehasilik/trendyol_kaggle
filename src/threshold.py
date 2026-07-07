"""Faz 5: threshold araclari — proxy sweep + rate matching + submission yazici.

Kullanim:
  python threshold.py <test_scores.npy> <proxy_scores.npz> [--rates 0.20,0.22,0.25]
Proxy-optimal esik + istenen pozitif oranlari veren esikler icin ayri ayri
submission dosyalari yazar (LB probing icin).
"""
import sys

import numpy as np
import pandas as pd

import config as C
import eval_proxy


def main():
    scores_path, proxy_path = sys.argv[1], sys.argv[2]
    rates = [0.20, 0.22, 0.25]
    for a in sys.argv[3:]:
        if a.startswith("--rates"):
            rates = [float(x) for x in a.split("=")[1].split(",")]

    scores = np.load(scores_path)
    pz = np.load(proxy_path, allow_pickle=True)
    thr_p, f1_p, _ = eval_proxy.sweep_threshold(pz["y"], pz["scores"])
    print(f"proxy-optimal esik: {thr_p:.2f} (proxy f1={f1_p:.4f})")

    sub_ids = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["id"], dtype=str)["id"]
    stem = scores_path.split("\\")[-1].split("/")[-1].replace(".npy", "")

    variants = {f"proxyopt{thr_p:.2f}": thr_p}
    qs = np.quantile(scores, [1 - r for r in rates])
    for r, q in zip(rates, qs):
        variants[f"rate{int(r*100)}"] = float(q)

    for name, thr in variants.items():
        pred = (scores > thr).astype(int)
        out = C.OUTPUT_DIR / f"sub_{stem}_{name}.csv"
        pd.DataFrame({"id": sub_ids, "prediction": pred}).to_csv(out, index=False)
        print(f"  {name:<14} thr={thr:.4f}  pozitif={pred.mean():.3f}  -> {out.name}")


if __name__ == "__main__":
    main()
