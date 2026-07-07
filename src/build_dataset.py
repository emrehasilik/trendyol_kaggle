"""Faz 1b (revize): test-negatif dagilimina oturtulmus negatif uretimi.

Audit bulgusu: bi-encoder ANN top-200 komsulari (sim ~0.75-0.90) buyuk olasilikla
GERCEK POZITIF (katalogda sorgu basina yuzlerce alakali urun var). Gercek test
adaylari ise cok daha dusuk sim bandinda (ort 0.35). Bu yuzden:
  - ANN top-200 uyeleri negatif havuzundan TAMAMEN YASAKLANIR (false-neg korumasi).
  - Negatif adaylar {pozitiflerin yaprak kategorisi %50, ust kategori %25,
    rastgele %25} havuzlarindan cekilir.
  - Adaylarin embed_sim'i GPU'da hesaplanir; testin TAHMINI NEGATIF histogramina
    (test_hist - POS_RATE_TEST * pos_hist) bin-bazli agirlikli orneklemeyle oturtulur.
  - Ayrica per-term p25 guard'ini gecen az sayidaki ultra-hard ANN adayi da
    havuza eklenir ('annhard' tier) - histogram agirligi ne kadarini alacagini belirler.

Ayni uretecle iki cikti:
  artifacts/train_dataset.parquet  (proxy disi termler; term_id,item_id,label,tier)
  artifacts/proxy_lists.parquet    (2000 proxy term; test benzeri ~100'luk listeler)
"""
import time

import numpy as np
import pandas as pd
import torch

import config as C

POS_RATE_TEST = 0.22          # eski submission'in pozitif orani (tahmin)
NEG_PER_POS = 4               # k pozitif icin 4k negatif hedefi (%20 pozitif)
OVERSAMPLE = 2.5              # bin esleme icin aday fazlasi
SIM_CEIL = 0.78               # mutlak guvenlik tavani
BINS = np.arange(-0.3, 1.0001, 0.05)


def leaf_of(cat):
    return cat.rsplit("/", 1)[-1].strip() if cat else ""


def top_of(cat):
    return cat.split("/", 1)[0].strip() if cat else ""


def gpu_sims(t_rows, i_rows, term_emb, item_gpu, device):
    """2M+ cift icin embed_sim, GPU'da chunkli."""
    n = len(t_rows)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, 100_000):
        e = min(s + 100_000, n)
        q = torch.from_numpy(term_emb[t_rows[s:e]]).to(device, dtype=torch.float16)
        d = item_gpu.index_select(0, torch.from_numpy(i_rows[s:e]).to(device))
        out[s:e] = (q * d).sum(dim=1).float().cpu().numpy()
        del q, d
    return out


def target_neg_hist(pos_sims):
    """Testin tahmini negatif-sim histogrami (karisimdan pozitifi cikar)."""
    test_sims = np.load(C.ARTIFACTS_DIR / "audit_test_sims.npy")
    h_test, _ = np.histogram(test_sims, bins=BINS, density=False)
    h_pos, _ = np.histogram(pos_sims, bins=BINS, density=False)
    h_test = h_test / h_test.sum()
    h_pos = h_pos / h_pos.sum()
    h_neg = np.clip(h_test - POS_RATE_TEST * h_pos, 0, None) / (1 - POS_RATE_TEST)
    return h_neg / h_neg.sum()


def main():
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    device = "cuda"

    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)
    items_meta = pd.read_csv(C.ITEMS_CSV, usecols=["item_id", "category"], dtype=str,
                             keep_default_na=False)
    item_ids = items_meta["item_id"].values
    i_index = {v: i for i, v in enumerate(item_ids)}
    n_items = len(item_ids)

    leaf = items_meta["category"].map(leaf_of).values
    top = items_meta["category"].map(top_of).values
    leaf_pool = pd.Series(np.arange(n_items)).groupby(leaf).apply(np.asarray).to_dict()
    top_pool = pd.Series(np.arange(n_items)).groupby(top).apply(np.asarray).to_dict()

    ann = np.load(C.ARTIFACTS_DIR / "ann_train.npz", allow_pickle=True)
    ann_idx, ann_sim = ann["idx"], ann["sim"].astype(np.float32)
    ann_row = {t: i for i, t in enumerate(ann["term_ids"])}

    term_ids_all = pd.read_csv(C.TERMS_CSV, usecols=["term_id"], dtype=str)["term_id"].values
    t_index = {v: i for i, v in enumerate(term_ids_all)}
    term_emb = np.load(C.TERM_EMB_NPY)

    # item matrisi GPU'ya (fp16, ~740 MB)
    mm = np.load(C.ITEM_EMB_NPY, mmap_mode="r")
    item_gpu = torch.empty((n_items, 384), dtype=torch.float16, device=device)
    for s in range(0, n_items, 100_000):
        e = min(s + 100_000, n_items)
        item_gpu[s:e] = torch.from_numpy(np.asarray(mm[s:e])).to(device, dtype=torch.float16)
    print(f"hazirlik tamam ({time.time()-t0:.0f}s)")

    # ---- proxy split
    uniq_terms = train["term_id"].drop_duplicates().values
    proxy_terms = rng.choice(uniq_terms, size=C.N_PROXY_TERMS, replace=False)
    np.save(C.ARTIFACTS_DIR / "proxy_terms.npy", proxy_terms)
    proxy_set = set(proxy_terms)

    # ---- pozitif simler (guard + hedef histogram icin)
    train["t_row"] = train["term_id"].map(t_index).astype(np.int64)
    train["i_row"] = train["item_id"].map(i_index).astype(np.int64)
    train["pos_sim"] = gpu_sims(train["t_row"].values, train["i_row"].values,
                                term_emb, item_gpu, device)
    h_target = target_neg_hist(train["pos_sim"].values)
    print("hedef negatif histogrami:",
          " ".join(f"{b:.2f}:{v:.3f}" for b, v in zip(BINS[:-1], h_target) if v > 0.01))

    # ---- aday uretimi (tum termler; proxy dahil - ciktilar sonra ayrilir)
    grouped = train.groupby("term_id", sort=False)
    cand_term_rows, cand_item_rows, cand_tier, cand_term_id, cand_need = [], [], [], [], []
    pos_records = []

    for term_id, g in grouped:
        pos_rows = g["i_row"].values
        k = len(pos_rows)
        pos_set = set(pos_rows.tolist())
        t_row = g["t_row"].iloc[0]
        is_proxy = term_id in proxy_set

        if is_proxy:
            need = max(C.PROXY_LIST_SIZE - k, 0)
        else:
            need = NEG_PER_POS * k
            if k + need > C.MAX_LIST_PER_TERM and k < C.MAX_LIST_PER_TERM:
                need = C.MAX_LIST_PER_TERM - k
        pos_records.append((term_id, pos_rows, is_proxy))
        if need == 0:
            continue

        n_draw = int(need * OVERSAMPLE)
        n_leaf = n_draw // 2
        n_top = n_draw // 4
        n_rand = n_draw - n_leaf - n_top

        draws, tiers = [], []
        pos_leaves = list({leaf[r] for r in pos_rows})
        pools = [leaf_pool[lv] for lv in pos_leaves if lv in leaf_pool]
        if pools:
            pool = pools[0] if len(pools) == 1 else np.concatenate(pools)
            take = rng.choice(pool, size=min(n_leaf, len(pool)), replace=False)
            draws.append(take); tiers.append(np.full(len(take), "leaf", dtype=object))
        pos_tops = list({top[r] for r in pos_rows})
        tpools = [top_pool[tv] for tv in pos_tops if tv in top_pool]
        if tpools:
            tpool = tpools[0] if len(tpools) == 1 else np.concatenate(tpools)
            take = rng.choice(tpool, size=min(n_top, len(tpool)), replace=False)
            draws.append(take); tiers.append(np.full(len(take), "top", dtype=object))
        take = rng.integers(0, n_items, size=n_rand)
        draws.append(take); tiers.append(np.full(len(take), "rand", dtype=object))

        # ultra-hard ANN bonusu: per-term p25 guard'ini gecenler
        a_row = ann_row.get(term_id)
        ann_members = set()
        if a_row is not None:
            ann_members = set(ann_idx[a_row].tolist())
            sim_cut = np.percentile(g["pos_sim"].values, C.POS_SIM_PERCENTILE)
            hard_mask = ann_sim[a_row, C.ANN_SKIP_TOP:] < sim_cut
            hard_pool = ann_idx[a_row, C.ANN_SKIP_TOP:][hard_mask]
            hard_pool = np.array([c for c in hard_pool if c not in pos_set], dtype=np.int64)
            if len(hard_pool):
                draws.append(hard_pool)
                tiers.append(np.full(len(hard_pool), "annhard", dtype=object))

        d = np.concatenate(draws).astype(np.int64)
        tr = np.concatenate(tiers)
        # pozitifler ve ANN-200 uyeleri (annhard haric) yasak
        keep = np.array([(c not in pos_set) and (tt == "annhard" or c not in ann_members)
                         for c, tt in zip(d, tr)])
        d, tr = d[keep], tr[keep]
        if len(d) == 0:
            continue
        cand_term_rows.append(np.full(len(d), t_row, dtype=np.int64))
        cand_item_rows.append(d)
        cand_tier.append(tr)
        cand_term_id.append(np.full(len(d), term_id, dtype=object))
        cand_need.append((term_id, need))

    ct = np.concatenate(cand_term_rows)
    ci = np.concatenate(cand_item_rows)
    ctier = np.concatenate(cand_tier)
    ctid = np.concatenate(cand_term_id)
    need_map = dict(cand_need)
    print(f"aday havuzu: {len(ci)} cift  ({time.time()-t0:.0f}s)")

    sims = gpu_sims(ct, ci, term_emb, item_gpu, device)
    ceil_keep = sims < SIM_CEIL
    ct, ci, ctier, ctid, sims = ct[ceil_keep], ci[ceil_keep], ctier[ceil_keep], ctid[ceil_keep], sims[ceil_keep]

    # ---- bin-bazli global agirliklar
    bin_ix = np.clip(np.digitize(sims, BINS) - 1, 0, len(BINS) - 2)
    h_avail = np.bincount(bin_ix, minlength=len(BINS) - 1).astype(np.float64)
    h_avail_frac = h_avail / h_avail.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        w_bin = np.where(h_avail_frac > 0, h_target / h_avail_frac, 0.0)
    weights = w_bin[bin_ix]
    print(f"tavan sonrasi aday={len(ci)}, agirliklar hazir  ({time.time()-t0:.0f}s)")

    # ---- per-term agirlikli secim
    order = np.argsort(ctid, kind="stable")
    ct, ci, ctier, ctid, sims, weights = (a[order] for a in (ct, ci, ctier, ctid, sims, weights))
    bounds = np.flatnonzero(np.r_[True, ctid[1:] != ctid[:-1], True])

    sel_indices = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        term_id = ctid[b0]
        need = need_map.get(term_id, 0)
        m = b1 - b0
        if need == 0 or m == 0:
            continue
        w = weights[b0:b1]
        if w.sum() <= 0:
            w = np.ones(m)
        take = min(need, m)
        p = w / w.sum()
        # agirlikli, tekrar cekimsiz secim
        pick = rng.choice(m, size=take, replace=False, p=p)
        sel_indices.append(b0 + pick)

    sel = np.concatenate(sel_indices)
    neg_df = pd.DataFrame({
        "term_id": ctid[sel],
        "i_row": ci[sel],
        "label": np.zeros(len(sel), dtype=np.int8),
        "tier": ctier[sel],
        "sim": sims[sel],
    })

    pos_df = pd.DataFrame({
        "term_id": train["term_id"],
        "i_row": train["i_row"],
        "label": np.ones(len(train), dtype=np.int8),
        "tier": "pos",
        "sim": train["pos_sim"],
    })

    full = pd.concat([pos_df, neg_df], ignore_index=True)
    full["item_id"] = item_ids[full["i_row"].values]
    full["is_proxy"] = full["term_id"].isin(proxy_set)

    ds = full[~full["is_proxy"]].sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    proxy = full[full["is_proxy"]].reset_index(drop=True)

    print(f"train dataset: {len(ds)} satir, pozitif={ds['label'].mean():.3f}")
    print(ds["tier"].value_counts().to_string())
    print(f"proxy listeleri: {len(proxy)} satir, {proxy['term_id'].nunique()} term, "
          f"pozitif={proxy['label'].mean():.3f}")

    # dagilim dogrulama
    test_sims = np.load(C.ARTIFACTS_DIR / "audit_test_sims.npy")
    pcts = [5, 25, 50, 75, 95]
    q_ds = np.percentile(ds["sim"].values, pcts)
    q_te = np.percentile(test_sims, pcts)
    print("dagilim (dataset tum satirlar vs test):")
    print("  dataset mean=%.3f  %s" % (ds["sim"].mean(), " ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, q_ds))))
    print("  test    mean=%.3f  %s" % (test_sims.mean(), " ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, q_te))))

    cols = ["term_id", "item_id", "label", "tier", "sim"]
    ds[cols].to_parquet(C.ARTIFACTS_DIR / "train_dataset.parquet", index=False)
    proxy[cols].to_parquet(C.ARTIFACTS_DIR / "proxy_lists.parquet", index=False)
    print(f"kaydedildi ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
