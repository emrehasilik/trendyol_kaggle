"""LB-proxy degerlendirme yardimcilari.

Proxy listeleri (artifacts/proxy_lists.parquet) test yapisini taklit eder:
2000 term x ~100 aday. Etiketlenmemis gercek pozitifler negatif sayildigi icin
skor KARAMSARDIR; onemli olan LB ile birlikte hareket etmesi.
"""
import numpy as np
from sklearn.metrics import f1_score


def sweep_threshold(labels, scores, lo=0.05, hi=0.95, step=0.01):
    """En iyi macro-F1 veren esigi bul. (best_thr, best_f1, f1_at_05) dondurur."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(labels, (scores > thr).astype(int), average="macro")
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    f1_05 = f1_score(labels, (scores > 0.5).astype(int), average="macro")
    return best_thr, best_f1, float(f1_05)


def report(name, labels, scores):
    thr, f1, f1_05 = sweep_threshold(labels, scores)
    pos_rate = float((scores > thr).mean())
    print(f"[{name}] macro_f1@0.5={f1_05:.4f}  en_iyi_thr={thr:.2f} f1={f1:.4f}  "
          f"pozitif_orani@thr={pos_rate:.3f}")
    return {"name": name, "f1_05": f1_05, "best_thr": thr, "best_f1": f1,
            "pos_rate": pos_rate}
