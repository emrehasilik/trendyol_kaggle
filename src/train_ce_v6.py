"""v6: buyuk govde cross-encoder egitimi (Colab/A100-L4 hedefli, adim-bazli resumable).

Karar gerekcesi HANDOFF.md bolum 7'de. Ozet:
- Govde: BAAI/bge-reranker-v2-m3 (XLM-R-large, sorgu-pasaj alakasi icin on-egitimli;
  num_labels=1 basligi HAZIR geliyor -> sicak baslangic). TK2_MODEL ile degistirilebilir.
- Veri: train_dataset_v5.parquet (v4 uzlasma etiketleri + proxy'nin gercek pozitifleri).
- TEK model, TUM veri (3-fold YOK): proxy pusulaliktan emekli, OOF cogunlukla pseudo
  uzerinde oldugundan metrik degil; fold ayirmak %33 veri kaybi + 3x inference demek.
  Izleme icin %2 terim holdout (NaN/raydan-cikma gozculugu — model SECIMI yapilmaz).
- XLM-R/DeBERTa token_type_ids KULLANMAZ; cift sablonu tokenizer'dan probe ile
  cikarilir (probe_pair_template — model-agnostik, transformers v4/v5 uyumlu).
- Resumable: TK2_CKPT_MIN dakikada bir (vars. 20) model fp16 + global_step Drive'a
  atomik yazilir. Oturum kopunca AYNI komut kaldigi yerden surer (optimizer durumu
  kaydedilmez; 200 adimlik lr yeniden-isinmasi ile telafi — kayip ihmal edilebilir).
- Hassasiyet: compute capability >= 8 (A100/L4) -> bf16; degilse fp16+GradScaler
  (T4'te mdeberta'dan KACIN — HANDOFF donanim dersi).

Kullanim: python train_ce_v6.py [--data train_dataset_v5.parquet] [--tag v6]
Cikti   : models/ce_{tag}_final.pt (fp16 sd), models/ce_{tag}_state.pt (ara durum),
          artifacts/ce_{tag}_holdout.npz, artifacts/ce_{tag}_proxy_scores.npz (varsa)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as C
from tokenize_ce_cache_v6 import tokens_path

DEVICE = "cuda"
REWARM_STEPS = 200      # resume sonrasi lr yeniden-isinma
SMOKE_AT = 50           # ilk olcum adimi (hiz+VRAM+NaN) — HANDOFF donanim dersi
NAN_LIMIT = 50          # ust uste bu kadar non-finite loss -> egitimi durdur
REAL_SOURCES = ("pos", "annvet", "v1neg", "proxy_pos")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def probe_pair_template(tok):
    """Tokenizer'in cift sablonunu (prefix/mid/suffix) yalniz __call__ ile cikarir.

    build_inputs_with_special_tokens transformers v5'te kaldirildi; bu yol her
    surumde calisir: iki bilinen kelime cift olarak encode edilir, ozel tokenlar
    konumlarindan ayristirilir (XLM-R: <s> A </s></s> B </s>).
    """
    a = tok("sol", add_special_tokens=False)["input_ids"]
    b = tok("sag", add_special_tokens=False)["input_ids"]
    pair = tok("sol", "sag")["input_ids"]

    def find(hay, needle, start):
        for k in range(start, len(hay) - len(needle) + 1):
            if hay[k:k + len(needle)] == needle:
                return k
        raise ValueError("cift sablonu cikarilamadi (probe eslesmedi)")

    ia = find(pair, a, 0)
    ib = find(pair, b, ia + len(a))
    return pair[:ia], pair[ia + len(a):ib], pair[ib + len(b):]


class TokenCacheV6:
    """Term/item token dizileri + model-agnostik cift sablonu (prefix/mid/suffix)."""

    def __init__(self):
        t = np.load(tokens_path("terms"))
        self.t_vals, self.t_off = t["vals"], t["off"]
        i = np.load(tokens_path("items"))
        self.i_vals, self.i_off = i["vals"], i["off"]

        tok = AutoTokenizer.from_pretrained(C.CE_V6_MODEL_NAME, cache_dir=C.HF_CACHE)
        self.pad_id = tok.pad_token_id
        pre, mid, suf = probe_pair_template(tok)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.mid = np.asarray(mid, dtype=np.int64)
        self.suf = np.asarray(suf, dtype=np.int64)
        self.overhead = len(self.pre) + len(self.mid) + len(self.suf)

        terms = pd.read_csv(C.TERMS_CSV, usecols=["term_id"], dtype=str)["term_id"].values
        items = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
        self.t_index = {v: k for k, v in enumerate(terms)}
        self.i_index = {v: k for k, v in enumerate(items)}

    def pair_len(self, t_rows, i_rows):
        return (self.t_off[t_rows + 1] - self.t_off[t_rows]) + \
               (self.i_off[i_rows + 1] - self.i_off[i_rows])


class PairDatasetV6(Dataset):
    def __init__(self, t_rows, i_rows, labels, cache: TokenCacheV6):
        self.t_rows, self.i_rows = t_rows, i_rows
        self.labels = labels
        self.c = cache

    def __len__(self):
        return len(self.t_rows)

    def __getitem__(self, idx):
        c = self.c
        tr, ir = self.t_rows[idx], self.i_rows[idx]
        t = c.t_vals[c.t_off[tr]:c.t_off[tr + 1]].astype(np.int64)
        it = c.i_vals[c.i_off[ir]:c.i_off[ir + 1]].astype(np.int64)
        avail = C.CE_V6_MAX_LENGTH - c.overhead - len(t)
        it = it[:max(avail, 0)]
        ids = np.concatenate([c.pre, t, c.mid, it, c.suf])
        y = self.labels[idx] if self.labels is not None else 0.0
        return ids, np.float32(y)


def collate_v6(batch, pad_id):
    maxlen = max(len(b[0]) for b in batch)
    n = len(batch)
    ids = np.full((n, maxlen), pad_id, dtype=np.int64)
    am = np.zeros((n, maxlen), dtype=np.int64)
    ys = np.empty(n, dtype=np.float32)
    for k, (i, y) in enumerate(batch):
        ids[k, :len(i)] = i
        am[k, :len(i)] = 1
        ys[k] = y
    return torch.from_numpy(ids), torch.from_numpy(am), torch.from_numpy(ys)


def amp_setup():
    """(autocast dtype, GradScaler|None). Ampere+ -> bf16, gerisi fp16+scaler."""
    if torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def auto_batch():
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gb >= 35:
        bs = 64          # A100
    elif gb >= 20:
        bs = 32          # L4
    else:
        bs = 16          # T4 vb.
    return bs, C.CE_V6_EFF_BATCH // bs


def score_pairs_v6(model, t_rows, i_rows, cache, amp_dtype, desc=""):
    """Uzunluga gore siralayip dinamik padding ile skorlar; orijinal siraya doner."""
    bs = C.CE_V6_INFER_BATCH
    order = np.argsort(cache.pair_len(t_rows, i_rows), kind="stable")
    ds = PairDatasetV6(t_rows[order], i_rows[order], None, cache)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2,
                    collate_fn=lambda b: collate_v6(b, cache.pad_id), pin_memory=True)
    model.eval()
    outs = np.empty(len(t_rows), dtype=np.float32)
    pos = 0
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        for ids, am, _ in dl:
            logits = model(input_ids=ids.to(DEVICE, non_blocking=True),
                           attention_mask=am.to(DEVICE, non_blocking=True)
                           ).logits.squeeze(-1)
            outs[pos:pos + len(ids)] = torch.sigmoid(logits.float()).cpu().numpy()
            pos += len(ids)
            if pos % (bs * 200) < bs:
                print(f"    {desc} {pos}/{len(t_rows)} ({time.time()-t0:.0f}s)",
                      flush=True)
    model.train()
    unsorted = np.empty_like(outs)
    unsorted[order] = outs
    return unsorted


def lr_at(gstep, total_steps, warmup, resume_at):
    if gstep < warmup:
        lr = C.CE_V6_LR * (gstep + 1) / max(1, warmup)
    else:
        p = (gstep - warmup) / max(1, total_steps - warmup)
        lr = C.CE_V6_LR * max(0.0, 1.0 - p)
    if resume_at > 0 and gstep - resume_at < REWARM_STEPS:  # taze optimizer isinmasi
        lr *= 0.1 + 0.9 * (gstep - resume_at) / REWARM_STEPS
    return lr


def evaluate_holdout(model, cache, hold, amp_dtype, tag, save=False):
    from sklearn.metrics import f1_score
    t_rows, i_rows, labels, sources = hold
    s = score_pairs_v6(model, t_rows, i_rows, cache, amp_dtype, desc="holdout")
    eps = 1e-7
    bce = float(-(labels * np.log(s + eps) + (1 - labels) * np.log(1 - s + eps)).mean())
    rm = np.isin(sources, REAL_SOURCES)
    f1 = f1_score(labels[rm], (s[rm] > 0.5).astype(int), average="macro")
    agree = float(((s[~rm] > 0.5) == (labels[~rm] > 0.5)).mean()) if (~rm).any() else 1.0
    print(f"[holdout] bce={bce:.4f}  macro_f1@0.5(gercek)={f1:.4f}  "
          f"pseudo-uyum={agree:.4f}  poz_oran@0.5={(s > 0.5).mean():.3f}", flush=True)
    if save:
        np.savez(C.ARTIFACTS_DIR / f"ce_{tag}_holdout.npz",
                 scores=s, y=labels, source=sources)


def save_state(path, model, gstep, bs, accum, n_rows):
    tmp = str(path) + ".tmp"
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                "gstep": gstep, "bs": bs, "accum": accum, "n_rows": n_rows}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_dataset_v5.parquet")
    ap.add_argument("--tag", default="v6")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    cache = TokenCacheV6()
    ds = pd.read_parquet(C.ARTIFACTS_DIR / args.data)
    t_rows = ds["term_id"].map(cache.t_index).values.astype(np.int64)
    i_rows = ds["item_id"].map(cache.i_index).values.astype(np.int64)
    labels = ds["label"].values.astype(np.float32)
    sources = ds["source"].values.astype(str)

    # %2 terim holdout (deterministik) — yalniz izleme
    uniq = np.sort(ds["term_id"].unique())
    hold_terms = set(np.random.default_rng(C.SEED).choice(
        uniq, size=int(len(uniq) * C.CE_V6_HOLDOUT_FRAC), replace=False))
    hmask = ds["term_id"].isin(hold_terms).values
    tr_idx = np.where(~hmask)[0]
    hold = (t_rows[hmask], i_rows[hmask], labels[hmask], sources[hmask])
    print(f"veri: {len(ds)} satir (poz={labels.mean():.3f})  "
          f"egitim={len(tr_idx)}  holdout={hmask.sum()} ({len(hold_terms)} terim)")

    amp_dtype, scaler = amp_setup()
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  amp={amp_dtype}  model={C.CE_V6_MODEL_NAME}")
    if amp_dtype == torch.float16 and "deberta" in C.CE_V6_MODEL_NAME.lower():
        raise SystemExit("mDeBERTa fp16'da NaN'a meyilli (HANDOFF dersi). "
                         "bf16 destekli GPU (A100/L4) sec ya da TK2_MODEL degistir.")

    model = AutoModelForSequenceClassification.from_pretrained(
        C.CE_V6_MODEL_NAME, num_labels=1, cache_dir=C.HF_CACHE).to(DEVICE)
    if os.environ.get("TK2_GRAD_CKPT") == "1":
        model.gradient_checkpointing_enable()

    final_path = C.MODELS_DIR / f"ce_{args.tag}_final.pt"
    state_path = C.MODELS_DIR / f"ce_{args.tag}_state.pt"

    if final_path.exists():
        print(f"{final_path.name} mevcut — egitim atlaniyor, holdout raporlaniyor")
        model.load_state_dict(torch.load(final_path, map_location=DEVICE,
                                         weights_only=True))
        evaluate_holdout(model, cache, hold, amp_dtype, args.tag, save=True)
        return

    gstep, resume_at = 0, 0
    if state_path.exists():
        st = torch.load(state_path, map_location=DEVICE, weights_only=True)
        assert st["n_rows"] == len(ds), "state farkli bir datasete ait!"
        model.load_state_dict(st["model"])
        gstep = resume_at = st["gstep"]
        bs, accum = st["bs"], st["accum"]  # adim matematigi bozulmasin
        print(f"RESUME: global_step={gstep} (bs={bs} accum={accum} state'ten)")
    else:
        bs, accum = auto_batch()

    steps_per_epoch = len(tr_idx) // (bs * accum)
    total_steps = steps_per_epoch * C.CE_V6_EPOCHS
    warmup = int(total_steps * C.CE_V6_WARMUP_RATIO)
    n_used = steps_per_epoch * bs * accum
    print(f"bs={bs} x accum={accum} (efektif {bs*accum})  "
          f"epoch_adim={steps_per_epoch}  toplam_adim={total_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=C.CE_V6_LR,
                            weight_decay=C.CE_V6_WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    t0 = time.time()
    last_save = time.time()
    nan_run = 0
    running, nrun, micro_seen = 0.0, 0, 0

    start_epoch = gstep // steps_per_epoch
    for epoch in range(start_epoch, C.CE_V6_EPOCHS):
        perm = np.random.default_rng(C.SEED * 1000 + epoch).permutation(tr_idx)[:n_used]
        done = (gstep - epoch * steps_per_epoch) * bs * accum  # epoch ici kaldigi yer
        perm = perm[done:]
        ds_tr = PairDatasetV6(t_rows[perm], i_rows[perm], labels[perm], cache)
        dl = DataLoader(ds_tr, batch_size=bs, shuffle=False, num_workers=2,
                        collate_fn=lambda b: collate_v6(b, cache.pad_id),
                        pin_memory=True, drop_last=True)
        print(f"epoch {epoch}: {len(ds_tr)} ornek kaldi (adim {gstep}/{total_steps})")

        for b, (ids, am, ys) in enumerate(dl):
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids=ids.to(DEVICE, non_blocking=True),
                               attention_mask=am.to(DEVICE, non_blocking=True)
                               ).logits.squeeze(-1)
                loss = loss_fn(logits.float(), ys.to(DEVICE)) / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            lval = loss.item() * accum
            if not np.isfinite(lval):
                nan_run += 1
                if nan_run >= NAN_LIMIT:
                    raise RuntimeError(
                        f"{NAN_LIMIT} ardisik non-finite loss — egitim raydan cikti. "
                        "lr'i dusur ya da bf16 GPU'ya gec.")
            else:
                nan_run = 0
                running += lval
                nrun += 1

            micro_seen += 1
            if micro_seen == SMOKE_AT:  # 50 adimlik olcum (HANDOFF donanim dersi)
                el = time.time() - t0
                vram = torch.cuda.max_memory_allocated() / 2**30
                eta_h = (total_steps - gstep) * accum * el / SMOKE_AT / 3600
                print(f"[smoke] {SMOKE_AT} micro-adim {el:.0f}s "
                      f"({SMOKE_AT/el:.1f} it/s)  VRAM={vram:.1f}GB  "
                      f"loss={running/max(nrun,1):.4f}  kalan~{eta_h:.1f}saat",
                      flush=True)

            if (b + 1) % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep, total_steps, warmup, resume_at)
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                if gstep % 500 == 0:
                    print(f"  e{epoch} adim {gstep}/{total_steps} "
                          f"loss={running/max(nrun,1):.4f} "
                          f"lr={opt.param_groups[0]['lr']:.2e} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    running, nrun = 0.0, 0

                if time.time() - last_save > C.CE_V6_CKPT_MINUTES * 60:
                    save_state(state_path, model, gstep, bs, accum, len(ds))
                    last_save = time.time()
                    print(f"  [ckpt] adim {gstep} kaydedildi "
                          f"({time.time()-t0:.0f}s)", flush=True)

        save_state(state_path, model, gstep, bs, accum, len(ds))
        last_save = time.time()
        print(f"epoch {epoch} bitti (adim {gstep})")
        evaluate_holdout(model, cache, hold, amp_dtype, args.tag)

    torch.save({k: v.half() for k, v in model.state_dict().items()}, final_path)
    print(f"egitim tamam: {final_path.name} ({time.time()-t0:.0f}s)")
    evaluate_holdout(model, cache, hold, amp_dtype, args.tag, save=True)

    # proxy sanity (SADECE felaket-kontrolu; kucuk farklarla karar VERME — DERS #2)
    proxy_path = C.ARTIFACTS_DIR / "proxy_lists.parquet"
    if proxy_path.exists():
        import eval_proxy
        proxy = pd.read_parquet(proxy_path)
        p_t = proxy["term_id"].map(cache.t_index).values.astype(np.int64)
        p_i = proxy["item_id"].map(cache.i_index).values.astype(np.int64)
        ps = score_pairs_v6(model, p_t, p_i, cache, amp_dtype, desc="proxy")
        np.savez(C.ARTIFACTS_DIR / f"ce_{args.tag}_proxy_scores.npz",
                 scores=ps, y=proxy["label"].values,
                 term_id=proxy["term_id"].values)
        eval_proxy.report(f"CE {args.tag} LB-proxy (sanity)",
                          proxy["label"].values, ps)
        print("UYARI: proxy yalniz felaket kontrolu — esik/karar LB'de verilir.")


if __name__ == "__main__":
    main()
