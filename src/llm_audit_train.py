"""v10 hazirlik: egitim SENTETIK negatiflerini LLM'e denetlet (false-negatif avi).

'Negatifler hatali mi?' sorusunun dogrudan testi. annvet + v1neg negatifleri
v1 soyunun sentetik uretimidir; icinde gercekte alakali (false-negatif) ciftler
olabilir ve bunlar modele "alakaliyi reddet" ogretir. LLM her cifti yargilar:
  p >= 0.7  -> anchor FALSE NEGATIF -> build_dataset_v10 pozitife cevirir (soft)
  0.3<p<0.7 -> kararsiz -> satir atilir
  p <= 0.3  -> onaylanmis negatif -> aynen kalir

Cikti : artifacts/llm_tr_{name}_idx.npy (v5 parquet SATIR indeksleri),
        artifacts/llm_tr_{name}_chunk{k:04d}.npz (20K'lik, resumable)
Kullanim:
  python llm_audit_train.py --limit 120            # goz kontrolu
  python llm_audit_train.py                        # tam kosu (resumable)
  python llm_audit_train.py --sources annvet,v1neg --name tr1
"""
import argparse
import time

import numpy as np
import pandas as pd

import config as C
from llm_judge_v8 import JCHUNK, make_llm, p_yes

BASE_PARQUET = "train_dataset_v5.parquet"


def tr_idx_path(name):
    return C.ARTIFACTS_DIR / f"llm_tr_{name}_idx.npy"


def tr_chunk_path(name, k):
    return C.ARTIFACTS_DIR / f"llm_tr_{name}_chunk{k:04d}.npz"


def load_train_audit(name):
    """Denetim sonuclari -> (v5 satir indeksleri, p)."""
    idx = np.load(tr_idx_path(name))
    n_chunks = (len(idx) + JCHUNK - 1) // JCHUNK
    missing = [k for k in range(n_chunks) if not tr_chunk_path(name, k).exists()]
    assert not missing, f"{name}: eksik chunk'lar {missing} — once denetimi bitir"
    parts = [np.load(tr_chunk_path(name, k)) for k in range(n_chunks)]
    jidx = np.concatenate([q["idx"] for q in parts])
    jp = np.concatenate([q["p"].astype(np.float32) for q in parts])
    assert np.array_equal(jidx, idx), f"{name}: chunk idx'leri secimle uyusmuyor"
    return jidx, jp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="annvet,v1neg")
    ap.add_argument("--name", default="tr1")
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    base = pd.read_parquet(C.ARTIFACTS_DIR / BASE_PARQUET)
    srcs = [s for s in args.sources.split(",") if s]
    mask = base["source"].isin(srcs).values & (base["label"].values < 0.5)

    ip = tr_idx_path(args.name)
    if ip.exists() and not args.limit:
        idx = np.load(ip)
        print(f"secim mevcut: {len(idx)} anchor ({ip.name})")
    else:
        idx = np.where(mask)[0].astype(np.int64)
        if len(idx) > args.cap:
            rng = np.random.default_rng(C.SEED)
            idx = np.sort(rng.choice(idx, size=args.cap, replace=False))
        print(f"denetlenecek anchor ({'+'.join(srcs)}): {len(idx)}")
        if not args.limit:
            np.save(ip, idx)
    if args.limit:
        rng = np.random.default_rng(C.SEED)
        pick = rng.choice(len(idx), size=min(args.limit, len(idx)), replace=False)
        idx = idx[np.sort(pick)]

    t_ids = base["term_id"].values[idx]
    i_ids = base["item_id"].values[idx]
    from llm_judge_v8 import pair_texts_from_ids
    queries, items = pair_texts_from_ids(t_ids, i_ids)
    print(f"metinler hazir ({time.time() - t0:.0f}s); LLM yukleniyor")
    llm, build, sp = make_llm()

    n = len(idx)
    n_chunks = (n + JCHUNK - 1) // JCHUNK
    flip_tot = 0
    for k in range(n_chunks):
        cp = tr_chunk_path(args.name, k)
        if not args.limit and cp.exists():
            print(f"chunk {k + 1}/{n_chunks} mevcut, atlaniyor")
            continue
        sl = slice(k * JCHUNK, min((k + 1) * JCHUNK, n))
        prompts = [build(q, it) for q, it in zip(queries[sl], items[sl])]
        outs = llm.generate(prompts, sp)
        p = np.array([p_yes(o.outputs[0].logprobs[0] if o.outputs[0].logprobs
                            else None, o.outputs[0].text) for o in outs],
                     dtype=np.float32)
        if args.limit:
            for j in range(min(15, len(p))):
                g = sl.start + j
                print(f"  llm={p[j]:.2f} | {queries[g][:40]!r} ~ {items[g][:70]!r}")
            print(f"[deneme] {len(p)} anchor; P(evet)>=0.7 orani="
                  f"{float((p >= .7).mean()):.3f} (false-neg adayi)")
            return
        np.savez(cp, idx=idx[sl], p=p.astype(np.float16))
        flip_tot += int((p >= 0.7).sum())
        el = time.time() - t0
        print(f"chunk {k + 1}/{n_chunks} bitti  {sl.stop}/{n}  "
              f"flip-aday={flip_tot}  "
              f"({el / 60:.0f} dk gecti, ~{el / sl.stop * (n - sl.stop) / 60:.0f} dk kaldi)",
              flush=True)
    print(f"denetim tamam ({(time.time() - t0) / 60:.0f} dk)")


if __name__ == "__main__":
    main()
