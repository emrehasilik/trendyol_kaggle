"""v8: LLM hakem — v6'nin gri bolge ciftlerini yerel buyuk LLM ile yeniden etiketle.

TANI (HANDOFF bolum 8): v1'den v6'ya TUM modeller ayni etiket soyundan ogreniyor
(250K gercek pozitif + sentetik/pseudo negatifler). Kanit:
  - model sinifi atlamasi (110M BERTurk -> 568M bge-reranker): +0.004 (0.851->0.855)
  - yeni-ogretmen pseudo dongusu (v7): -0.003
  - mimari ensemble (v2+v3+v4): +0.000
Yani tavan MODEL degil ETIKET. Gri bolgede (skor 0.05-0.95, ~%7 = ~250K cift)
tum modeller AYNI hatalari yapiyor cunku negatif tanimi hep ayni sentetik/pseudo
kaynaktan geliyor. Yeni bilgi ancak bu soydan BAGIMSIZ bir kaynaktan gelir:
buyuk bir acik LLM'in dunya bilgisi (Turkce urun/sorgu semantigi).

Iki asama:
  judge : gri bolge ciftlerini secer (karar esigine yakin olan oncelikli),
          vLLM ile Evet/Hayir olasiligi cikarir. 20K'lik chunk'lar Drive'a
          yazilir -> resumable. `--limit N` ile dosya yazmadan ornek gosterir.
  merge : LLM olasiligini CE skoruyla harmanlar (alpha*llm + (1-alpha)*ce),
          SABIT %25 pozitif SAYISIYLA keser (DERS #4), submission yazar.

Kullanim (Colab, A100):
  python llm_judge_v8.py judge --tag v6 --limit 120   # once goz kontrolu
  python llm_judge_v8.py judge --tag v6               # tam kosu (resumable)
  python llm_judge_v8.py merge --tag v6 --alpha 0.7   # submission
Model TK2_LLM env ile secilir (vars. Qwen2.5-32B-Instruct-AWQ; hiz gerekirse
Qwen/Qwen2.5-14B-Instruct-AWQ).
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

import config as C

JCHUNK = 20_000
LLM_NAME = os.environ.get("TK2_LLM", "Qwen/Qwen2.5-32B-Instruct-AWQ")

SYSTEM = ('Türkçe e-ticaret arama kalitesi uzmanısın. Sana bir arama terimi ve '
          'bir ürün verilecek. Ürün, kullanıcının aradığı ürün tipiyle '
          'eşleşiyorsa ve terimdeki marka, model, cinsiyet, yaş, renk, beden '
          'gibi kısıtlarla çelişmiyorsa alakalıdır; aksesuar/yedek parça gibi '
          'sadece ilgili ama aranan şey olmayan ürünler alakasızdır. '
          'Sadece "Evet" veya "Hayır" yaz.')


def scores_path(tag):
    return C.ARTIFACTS_DIR / f"ce_{tag}_test.npy"


def idx_path(tag):
    return C.ARTIFACTS_DIR / f"llm_{tag}_idx.npy"


def chunk_path(tag, k):
    return C.ARTIFACTS_DIR / f"llm_{tag}_chunk{k:04d}.npz"


def select_gray(scores, lo, hi, cap, rate):
    """Gri bolge indeksleri; cap asilirsa karar esigine en yakinlar oncelikli."""
    idx = np.where((scores > lo) & (scores < hi))[0]
    if len(idx) > cap:
        thr = float(np.quantile(scores, 1.0 - rate))
        d = np.abs(scores[idx] - thr)
        idx = np.sort(idx[np.argsort(d, kind="stable")[:cap]])
    return idx.astype(np.int64)


def p_yes(first_logprobs, gen_text):
    """Ilk uretilen tokenin top-N logprob'undan P(Evet); olmazsa metinden."""
    py = pn = 0.0
    if first_logprobs:
        for lp in first_logprobs.values():
            t = (lp.decoded_token or "").replace("▁", "").strip().lower()
            if t.startswith("evet") or t in ("e", "ev", "eve"):
                py += math.exp(lp.logprob)
            elif t.startswith("hay") or t == "h":
                pn += math.exp(lp.logprob)
    if py + pn > 1e-9:
        return py / (py + pn)
    g = gen_text.strip().lower()
    return 1.0 if g.startswith("evet") else 0.0 if g.startswith("hay") else 0.5


def load_pair_texts(idx):
    """Secilen ciftler icin (sorgu, zengin item metni) listeleri."""
    from tokenize_ce_cache_v6 import item_text_v6
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)
    t_ids = sub["term_id"].values[idx]
    i_ids = sub["item_id"].values[idx]
    del sub
    terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
    q_of = dict(zip(terms["term_id"], terms["query"]))
    need = set(i_ids)
    it_of = {}
    for ch in pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                          chunksize=100_000):
        for r in ch[ch["item_id"].isin(need)].itertuples(index=False):
            it_of[r.item_id] = item_text_v6(r.title, r.brand, r.category,
                                            r.gender, r.age_group, r.attributes)
    return [q_of[t] for t in t_ids], [it_of[i] for i in i_ids]


def run_judge(args):
    t0 = time.time()
    s = np.load(scores_path(args.tag))
    ip = idx_path(args.tag)
    if ip.exists() and not args.limit:
        idx = np.load(ip)
        print(f"secim mevcut: {len(idx)} cift ({ip.name})")
    else:
        idx = select_gray(s, args.lo, args.hi, args.cap, args.rate)
        print(f"gri bolge ({args.lo}-{args.hi}): {len(idx)} cift "
              f"(testin %{100 * len(idx) / len(s):.1f}'i)")
        if not args.limit:
            np.save(ip, idx)
    if args.limit:
        rng = np.random.default_rng(C.SEED)
        pick = rng.choice(len(idx), size=min(args.limit, len(idx)), replace=False)
        idx = idx[np.sort(pick)]

    queries, items = load_pair_texts(idx)
    print(f"metinler hazir ({time.time() - t0:.0f}s); LLM yukleniyor: {LLM_NAME}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=LLM_NAME, max_model_len=1024, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, download_dir=C.HF_CACHE)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    def build(q, it):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f'Arama terimi: "{q}"\nÜrün: {it}\n\n'
                            f'Bu ürün bu arama için alakalı mı?'}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)

    n = len(idx)
    n_chunks = (n + JCHUNK - 1) // JCHUNK
    thr0 = float(np.quantile(s, 1.0 - args.rate))
    for k in range(n_chunks):
        cp = chunk_path(args.tag, k)
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
                print(f"  ce={s[idx[g]]:.2f} llm={p[j]:.2f} | "
                      f"{queries[g][:40]!r} ~ {items[g][:70]!r}")
            agree = float(((p >= .5) == (s[idx[sl]] > thr0)).mean())
            print(f"[deneme] {len(p)} cift; LLM-CE karar uyumu={agree:.2f}  "
                  f"ort P(evet)={p.mean():.2f}")
            return
        np.savez(cp, idx=idx[sl], p=p.astype(np.float16))
        el = time.time() - t0
        print(f"chunk {k + 1}/{n_chunks} bitti  {sl.stop}/{n}  "
              f"({el / 60:.0f} dk gecti, ~{el / sl.stop * (n - sl.stop) / 60:.0f} dk kaldi)",
              flush=True)
    print(f"judge tamam ({(time.time() - t0) / 60:.0f} dk)")


def run_merge(args):
    s = np.load(scores_path(args.tag))
    n = len(s)
    idx = np.load(idx_path(args.tag))
    n_chunks = (len(idx) + JCHUNK - 1) // JCHUNK
    missing = [k for k in range(n_chunks) if not chunk_path(args.tag, k).exists()]
    assert not missing, f"eksik chunk'lar: {missing} — once judge'i bitir"
    parts = [np.load(chunk_path(args.tag, k)) for k in range(n_chunks)]
    jidx = np.concatenate([q["idx"] for q in parts])
    jp = np.concatenate([q["p"].astype(np.float32) for q in parts])
    assert np.array_equal(jidx, idx), "chunk idx'leri secimle uyusmuyor"

    merged = s.copy()
    merged[idx] = args.alpha * jp + (1.0 - args.alpha) * s[idx]

    k = int(round(args.rate * n))

    def topk_pred(x):
        pr = np.zeros(n, dtype=np.int8)
        pr[np.argpartition(-x, k - 1)[:k]] = 1
        return pr

    pred, pred0 = topk_pred(merged), topk_pred(s)
    thr0 = float(np.quantile(s, 1.0 - args.rate))
    dis = float(((jp >= .5) != (s[idx] > thr0)).mean())
    changed = int((pred != pred0).sum())

    ids = pd.read_csv(C.SUBMISSION_PAIRS_CSV, usecols=["id"], dtype=str)["id"]
    out = (C.OUTPUT_DIR /
           f"sub_{args.tag}_llm{int(args.alpha * 100)}_rate{int(args.rate * 100)}.csv")
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)
    print(f"yargilanan={len(idx)}  LLM-CE karar farki=%{100 * dis:.1f}  "
          f"degisen tahmin={changed} (%{100 * changed / n:.2f})")
    print(f"yazildi: {out}  pozitif={int(pred.sum())} (oran {pred.mean():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["judge", "merge"])
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--rate", type=float, default=0.25)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    (run_judge if args.mode == "judge" else run_merge)(args)


if __name__ == "__main__":
    main()
