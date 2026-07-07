"""Faz 1a: her train termi icin katalogdaki en yakin ANN_TOP_K itemi bul (GPU).

Hazir bi-encoder embeddingleri kullanilir (satir sirasi = CSV sirasi).
Cikti: artifacts/ann_train.npz  (idx: int32 [n_terms, K], sim: float16 [n_terms, K],
                                 term_ids: train termlerin id listesi)

VRAM plani (6 GB): item matrisi fp16 GPU'da ~740 MB kalici; term chunk'i 512 icin
skor matrisi 512 x 962873 fp16 ~ 0.95 GB gecici. Toplam < 2.5 GB.
RAM plani (~2.6 GB bos): item_emb mmap'ten 100k satirlik dilimlerle GPU'ya tasinir.
"""
import time

import numpy as np
import pandas as pd
import torch

import config as C

TERM_CHUNK = 512
ITEM_COPY_CHUNK = 100_000


def load_item_matrix_gpu(device):
    mm = np.load(C.ITEM_EMB_NPY, mmap_mode="r")
    n, d = mm.shape
    gpu = torch.empty((n, d), dtype=torch.float16, device=device)
    for s in range(0, n, ITEM_COPY_CHUNK):
        e = min(s + ITEM_COPY_CHUNK, n)
        gpu[s:e] = torch.from_numpy(np.asarray(mm[s:e])).to(device, dtype=torch.float16)
    return gpu


def mine(term_ids, out_path):
    t0 = time.time()
    device = "cuda"
    all_term_ids = pd.read_csv(C.TERMS_CSV, usecols=["term_id"], dtype=str)["term_id"].values
    t_index = {v: i for i, v in enumerate(all_term_ids)}
    rows = np.fromiter((t_index[t] for t in term_ids), dtype=np.int64, count=len(term_ids))

    term_emb = np.load(C.TERM_EMB_NPY)[rows]  # [n, 384] float32
    item_gpu = load_item_matrix_gpu(device)   # [962873, 384] fp16
    print(f"item matrisi GPU'da ({time.time()-t0:.0f}s), mining basliyor: {len(rows)} term")

    n = len(rows)
    out_idx = np.empty((n, C.ANN_TOP_K), dtype=np.int32)
    out_sim = np.empty((n, C.ANN_TOP_K), dtype=np.float16)
    with torch.inference_mode():
        for s in range(0, n, TERM_CHUNK):
            e = min(s + TERM_CHUNK, n)
            q = torch.from_numpy(term_emb[s:e]).to(device, dtype=torch.float16)
            scores = q @ item_gpu.T                       # [chunk, 962873] fp16
            vals, idx = torch.topk(scores, C.ANN_TOP_K, dim=1)
            out_idx[s:e] = idx.cpu().numpy().astype(np.int32)
            out_sim[s:e] = vals.cpu().numpy().astype(np.float16)
            del q, scores, vals, idx
            if (s // TERM_CHUNK) % 10 == 0:
                print(f"  {e}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez(out_path, idx=out_idx, sim=out_sim, term_ids=np.asarray(term_ids))
    print(f"kaydedildi: {out_path}  ({time.time()-t0:.0f}s)")


def main():
    train_terms = pd.read_csv(C.TRAINING_PAIRS_CSV, usecols=["term_id"], dtype=str)["term_id"]
    uniq = train_terms.drop_duplicates().values
    mine(uniq, C.ARTIFACTS_DIR / "ann_train.npz")


if __name__ == "__main__":
    main()
