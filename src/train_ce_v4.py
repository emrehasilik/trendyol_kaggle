"""Faz 7: v4 = XLM-RoBERTa-base cross-encoder (mimari cesitliligi).

NOT: ilk v4 denemesi mDeBERTa idi - disentangled attention'in bmm ara
matrisleri 6GB karti tasirip Windows'un paylasimli RAM'ine sizdi (sessiz
10-20x yavaslama) -> XLM-R'a gecildi: ayni boyut sinifi, klasik attention,
bf16'da kararli.

BERTurk'ten farklar:
  - FacebookAI/xlm-roberta-base (250K SentencePiece vocab);
    kendi token cache'i: ce_tokens_*_xlmr.npz.
  - TEK model, TUM veriyle, 1 epoch (fold yok; degerlendirme proxy'de).
  - 6GB icin: batch 16 x accum 4 (efektif 64), gradient checkpointing.
  - XLM-R token_type embedding KULLANMAZ (type_vocab_size=1): modele
    token_type_ids GONDERILMEZ (gonderilirse index hatasi) -> score_pairs_v4.
  - nan zirhi: sonlu olmayan micro-batch/gradyan adimlari atlanir (skip sayaci).

Veri: train_dataset_v4.parquet (uzlasma-filtreli pseudo-labellar, v2+v3).
Cikti: models/ce_v4_full.pt, artifacts/ce_v4_proxy_scores.npz
"""
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config as C
import eval_proxy
from train_ce import PairDataset, collate

DEVICE = "cuda"
MODEL_NAME = "FacebookAI/xlm-roberta-base"
V4_DATASET = "train_dataset_v4.parquet"   # uzlasma-filtreli pseudo-labellar (v2+v3)
V4_BATCH = 16
V4_ACCUM = 4
V4_LR = 1e-5
V4_EPOCHS = 1
V4_WARMUP = 0.10


def score_pairs_v4(model, t_rows, i_rows, cache, desc=""):
    """train_ce.score_pairs'in token_type_ids gondermeyen kopyasi (XLM-R icin sart)."""
    bs = C.CE_INFER_BATCH_SIZE
    lens = (cache.t_off[t_rows + 1] - cache.t_off[t_rows]) + \
           (cache.i_off[i_rows + 1] - cache.i_off[i_rows])
    order = np.argsort(lens, kind="stable")
    sorted_ds = PairDataset(t_rows[order], i_rows[order], None, cache)
    dl = DataLoader(sorted_ds, batch_size=bs, shuffle=False, num_workers=0,
                    collate_fn=lambda b: collate(b, cache.pad_id), pin_memory=True)
    model.eval()
    outs = np.empty(len(t_rows), dtype=np.float32)
    pos = 0
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, tt, am, _ in dl:
            logits = model(input_ids=ids.to(DEVICE, non_blocking=True),
                           attention_mask=am.to(DEVICE, non_blocking=True)
                           ).logits.squeeze(-1)
            outs[pos:pos + len(ids)] = torch.sigmoid(logits.float()).cpu().numpy()
            pos += len(ids)
            if pos % (bs * 400) < bs:
                print(f"    {desc} {pos}/{len(t_rows)} ({time.time()-t0:.0f}s)", flush=True)
    unsorted = np.empty_like(outs)
    unsorted[order] = outs
    return unsorted


class TokenCacheM:
    """train_ce.TokenCache ile ayni arayuz, xlm-r cache'leriyle."""

    def __init__(self):
        t = np.load(C.ARTIFACTS_DIR / "ce_tokens_terms_xlmr.npz")
        self.t_vals, self.t_off = t["vals"], t["off"]
        i = np.load(C.ARTIFACTS_DIR / "ce_tokens_items_xlmr.npz")
        self.i_vals, self.i_off = i["vals"], i["off"]

        tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=C.HF_CACHE)
        self.cls_id = tok.cls_token_id
        self.sep_id = tok.sep_token_id
        self.pad_id = tok.pad_token_id

        terms = pd.read_csv(C.TERMS_CSV, usecols=["term_id"], dtype=str)["term_id"].values
        items = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
        self.t_index = {v: k for k, v in enumerate(terms)}
        self.i_index = {v: k for k, v in enumerate(items)}


def main():
    torch.manual_seed(C.SEED)
    t0 = time.time()
    cache = TokenCacheM()

    ds = pd.read_parquet(C.ARTIFACTS_DIR / V4_DATASET)
    t_rows = ds["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = ds["item_id"].map(cache.i_index).values.astype(np.int64)
    labels = ds["label"].values.astype(np.float32)
    print(f"veri: {V4_DATASET}  {len(ds)} satir  pozitif={labels.mean():.3f}")

    ckpt = C.MODELS_DIR / "ce_v4_full.pt"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)

    if ckpt.exists():
        print("checkpoint mevcut, egitim atlaniyor")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=True))
    else:
        model.gradient_checkpointing_enable()
        ds_tr = PairDataset(t_rows, i_rows, labels, cache)
        dl = DataLoader(ds_tr, batch_size=V4_BATCH, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, cache.pad_id),
                        pin_memory=True, drop_last=True)
        steps_per_epoch = len(dl) // V4_ACCUM
        total_steps = steps_per_epoch * V4_EPOCHS
        opt = torch.optim.AdamW(model.parameters(), lr=V4_LR,
                                weight_decay=C.CE_WEIGHT_DECAY)
        sched = get_linear_schedule_with_warmup(
            opt, int(total_steps * V4_WARMUP), total_steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        print(f"fold 0: {len(ds_tr)} ornek, {total_steps} optimizer adimi", flush=True)
        model.train()
        step = 0
        skipped = 0
        for epoch in range(V4_EPOCHS):
            running, nrun = 0.0, 0
            for b, (ids, tt, am, ys) in enumerate(dl):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids.to(DEVICE, non_blocking=True),
                                   attention_mask=am.to(DEVICE, non_blocking=True)
                                   ).logits.squeeze(-1)
                    loss = loss_fn(logits.float(), ys.to(DEVICE)) / V4_ACCUM
                # nan zirhi 1: bozuk micro-batch gradyani zehirlemeden atlanir
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                loss.backward()
                running += loss.item() * V4_ACCUM
                nrun += 1
                if (b + 1) % V4_ACCUM == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # nan zirhi 2: gradyan normu sonlu degilse adim atilmaz
                    if torch.isfinite(norm):
                        opt.step()
                        sched.step()
                    else:
                        skipped += 1
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 500 == 0:
                        print(f"  fold 0 e{epoch} step {step}/{total_steps} "
                              f"loss={running/max(nrun,1):.4f} skip={skipped} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                        running, nrun = 0.0, 0
        torch.save({k: v.half() for k, v in model.state_dict().items()}, ckpt)
        print(f"fold 0 egitildi ve kaydedildi ({time.time()-t0:.0f}s)", flush=True)
        model.gradient_checkpointing_disable()

    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    p_t = proxy["term_id"].map(cache.t_index).values.astype(np.int64)
    p_i = proxy["item_id"].map(cache.i_index).values.astype(np.int64)
    pr = score_pairs_v4(model, p_t, p_i, cache, desc="proxy f0")
    np.savez(C.ARTIFACTS_DIR / "ce_v4_proxy_scores.npz", scores=pr,
             y=proxy["label"].values, term_id=proxy["term_id"].values)

    eval_proxy.report("CE v4 LB-proxy", proxy["label"].values, pr)
    for tag, path in [("v2", "ce_v2_proxy_scores.npz"), ("v3", "ce_v3_proxy_scores.npz")]:
        p = C.ARTIFACTS_DIR / path
        if p.exists():
            z = np.load(p)
            eval_proxy.report(f"(kiyas) CE {tag} LB-proxy", z["y"], z["scores"])
            # ensemble on onizleme (proxy uzerinde esit agirlik)
            eval_proxy.report(f"(onizleme) v4+{tag} ensemble",
                              z["y"], (pr + z["scores"]) / 2)


if __name__ == "__main__":
    main()
