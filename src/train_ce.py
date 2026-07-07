"""Faz 3: cross-encoder egitimi (BERTurk 128k-uncased, 3-fold GroupKFold).

- Girdi: artifacts/train_dataset.parquet + ce token cacheleri.
- Fold basina 2 epoch, AdamW 2e-5, bf16 autocast, bs 32 x accum 2 (efektif 64).
- Fold sonunda: OOF skorlar + proxy skorlar + checkpoint (resumable: checkpoint
  varsa fold atlanir).
- Cikti: models/ce_fold{i}.pt, artifacts/ce_oof.npz, artifacts/ce_proxy_scores.npz
"""
import gc
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config as C
import eval_proxy

DEVICE = "cuda"


class TokenCache:
    def __init__(self):
        t = np.load(C.ARTIFACTS_DIR / "ce_tokens_terms.npz")
        self.t_vals, self.t_off = t["vals"], t["off"]
        i = np.load(C.ARTIFACTS_DIR / "ce_tokens_items.npz")
        self.i_vals, self.i_off = i["vals"], i["off"]

        tok = AutoTokenizer.from_pretrained(C.CE_MODEL_NAME, cache_dir=C.HF_CACHE)
        self.cls_id = tok.cls_token_id
        self.sep_id = tok.sep_token_id
        self.pad_id = tok.pad_token_id

        terms = pd.read_csv(C.TERMS_CSV, usecols=["term_id"], dtype=str)["term_id"].values
        items = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
        self.t_index = {v: k for k, v in enumerate(terms)}
        self.i_index = {v: k for k, v in enumerate(items)}


class PairDataset(Dataset):
    def __init__(self, t_rows, i_rows, labels, cache: TokenCache):
        self.t_rows, self.i_rows = t_rows, i_rows
        self.labels = labels
        self.c = cache

    def __len__(self):
        return len(self.t_rows)

    def __getitem__(self, idx):
        c = self.c
        tr, ir = self.t_rows[idx], self.i_rows[idx]
        t = c.t_vals[c.t_off[tr]:c.t_off[tr + 1]]
        it = c.i_vals[c.i_off[ir]:c.i_off[ir + 1]]
        avail = C.CE_MAX_LENGTH - 3 - len(t)
        it = it[:max(avail, 0)]
        ids = np.empty(len(t) + len(it) + 3, dtype=np.int64)
        ids[0] = c.cls_id
        ids[1:1 + len(t)] = t
        ids[1 + len(t)] = c.sep_id
        ids[2 + len(t):2 + len(t) + len(it)] = it
        ids[-1] = c.sep_id
        tt = np.zeros(len(ids), dtype=np.int64)
        tt[2 + len(t):] = 1
        y = self.labels[idx] if self.labels is not None else 0.0
        return ids, tt, np.float32(y)


def collate(batch, pad_id):
    maxlen = max(len(b[0]) for b in batch)
    n = len(batch)
    ids = np.full((n, maxlen), pad_id, dtype=np.int64)
    tt = np.zeros((n, maxlen), dtype=np.int64)
    am = np.zeros((n, maxlen), dtype=np.int64)
    ys = np.empty(n, dtype=np.float32)
    for k, (i, t, y) in enumerate(batch):
        L = len(i)
        ids[k, :L] = i
        tt[k, :L] = t
        am[k, :L] = 1
        ys[k] = y
    return (torch.from_numpy(ids), torch.from_numpy(tt),
            torch.from_numpy(am), torch.from_numpy(ys))


def score_pairs(model, t_rows, i_rows, cache, desc=""):
    """Uzunluga gore siralayip dinamik padding ile skorlar, orijinal siraya dondurur."""
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
                           token_type_ids=tt.to(DEVICE, non_blocking=True),
                           attention_mask=am.to(DEVICE, non_blocking=True)
                           ).logits.squeeze(-1)
            outs[pos:pos + len(ids)] = torch.sigmoid(logits.float()).cpu().numpy()
            pos += len(ids)
            if pos % (bs * 400) < bs:
                print(f"    {desc} {pos}/{len(t_rows)} ({time.time()-t0:.0f}s)", flush=True)
    unsorted = np.empty_like(outs)
    unsorted[order] = outs
    return unsorted


def train_fold(fold, tr_idx, data, cache):
    t_rows, i_rows, labels = data
    ckpt = C.MODELS_DIR / f"ce_fold{fold}.pt"

    model = AutoModelForSequenceClassification.from_pretrained(
        C.CE_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)

    if ckpt.exists():
        print(f"fold {fold}: checkpoint mevcut, egitim atlaniyor")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        return model

    ds_tr = PairDataset(t_rows[tr_idx], i_rows[tr_idx], labels[tr_idx], cache)
    dl = DataLoader(ds_tr, batch_size=C.CE_BATCH_SIZE, shuffle=True, num_workers=0,
                    collate_fn=lambda b: collate(b, cache.pad_id),
                    pin_memory=True, drop_last=True)

    steps_per_epoch = len(dl) // C.CE_GRAD_ACCUM
    total_steps = steps_per_epoch * C.CE_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=C.CE_LR,
                            weight_decay=C.CE_WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * C.CE_WARMUP_RATIO), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"fold {fold}: {len(ds_tr)} ornek, {total_steps} optimizer adimi")
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(C.CE_EPOCHS):
        running, nrun = 0.0, 0
        for b, (ids, tt, am, ys) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.to(DEVICE, non_blocking=True),
                               token_type_ids=tt.to(DEVICE, non_blocking=True),
                               attention_mask=am.to(DEVICE, non_blocking=True)
                               ).logits.squeeze(-1)
                loss = loss_fn(logits.float(), ys.to(DEVICE)) / C.CE_GRAD_ACCUM
            loss.backward()
            running += loss.item() * C.CE_GRAD_ACCUM
            nrun += 1
            if (b + 1) % C.CE_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 500 == 0:
                    print(f"  fold {fold} e{epoch} step {step}/{total_steps} "
                          f"loss={running/nrun:.4f} ({time.time()-t0:.0f}s)",
                          flush=True)
                    running, nrun = 0.0, 0
    # disk tasarrufu: fp16 kaydet (~370 MB/fold); inference zaten bf16 autocast
    torch.save({k: v.half() for k, v in model.state_dict().items()}, ckpt)
    print(f"fold {fold} egitildi ve kaydedildi ({time.time()-t0:.0f}s)")
    return model


def main():
    torch.manual_seed(C.SEED)
    cache = TokenCache()

    ds = pd.read_parquet(C.ARTIFACTS_DIR / "train_dataset.parquet")
    t_rows = ds["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = ds["item_id"].map(cache.i_index).values.astype(np.int64)
    labels = ds["label"].values.astype(np.float32)
    groups = ds["term_id"].values

    proxy = pd.read_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet")
    p_t = proxy["term_id"].map(cache.t_index).values.astype(np.int64)
    p_i = proxy["item_id"].map(cache.i_index).values.astype(np.int64)

    gkf = GroupKFold(n_splits=C.CE_N_FOLDS)
    oof = np.zeros(len(ds), dtype=np.float32)
    proxy_scores = np.zeros(len(proxy), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(t_rows, labels, groups)):
        model = train_fold(fold, tr_idx, (t_rows, i_rows, labels), cache)

        oof_path = C.ARTIFACTS_DIR / f"ce_oof_fold{fold}.npy"
        if oof_path.exists():
            oof[va_idx] = np.load(oof_path)
        else:
            oof[va_idx] = score_pairs(model, t_rows[va_idx], i_rows[va_idx],
                                      cache, desc=f"oof f{fold}")
            np.save(oof_path, oof[va_idx])
        f1 = f1_score(labels[va_idx], (oof[va_idx] > 0.5).astype(int), average="macro")
        print(f"fold {fold} OOF macro_f1@0.5 = {f1:.4f}")

        pr_path = C.ARTIFACTS_DIR / f"ce_proxy_fold{fold}.npy"
        if pr_path.exists():
            proxy_scores += np.load(pr_path) / C.CE_N_FOLDS
        else:
            pr = score_pairs(model, p_t, p_i, cache, desc=f"proxy f{fold}")
            np.save(pr_path, pr)
            proxy_scores += pr / C.CE_N_FOLDS

        del model
        gc.collect()
        torch.cuda.empty_cache()

    np.savez(C.ARTIFACTS_DIR / "ce_oof.npz", oof=oof, y=labels,
             term_id=ds["term_id"].values, item_id=ds["item_id"].values)
    np.savez(C.ARTIFACTS_DIR / "ce_proxy_scores.npz", scores=proxy_scores,
             y=proxy["label"].values, term_id=proxy["term_id"].values)

    eval_proxy.report("CE OOF (ornekli)", labels, oof)
    eval_proxy.report("CE LB-proxy", proxy["label"].values, proxy_scores)


if __name__ == "__main__":
    main()
