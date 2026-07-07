"""Faz 0 auditi: test ciftlerinin embed_sim dagilimi nerede duruyor?

Hipotez: test ciftleri retrieval aday listesi oldugu icin dagilim,
rastgele ciftler ile train pozitifleri ARASINDA olmali. Bu dagilim,
Faz 1'de uretilecek negatiflerin hedef dagilimidir.

RAM notu: item_emb (1.48 GB) mmap ile acilir, sadece gereken satirlar okunur.
"""
import numpy as np
import pandas as pd

import config as C

N_SAMPLE = 200_000
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def id_index(csv_path, col):
    ids = pd.read_csv(csv_path, usecols=[col], dtype=str)[col].values
    return {v: i for i, v in enumerate(ids)}


def pair_sims(term_ids, item_ids, t_index, i_index, term_emb, item_emb):
    t_rows = np.fromiter((t_index[t] for t in term_ids), dtype=np.int64, count=len(term_ids))
    i_rows = np.fromiter((i_index[i] for i in item_ids), dtype=np.int64, count=len(item_ids))
    sims = np.empty(len(t_rows), dtype=np.float32)
    for s in range(0, len(t_rows), 50_000):
        e = min(s + 50_000, len(t_rows))
        sims[s:e] = np.sum(term_emb[t_rows[s:e]] * item_emb[i_rows[s:e]], axis=1)
    return sims


def report(name, sims):
    q = np.percentile(sims, PCTS)
    line = " ".join(f"p{p}={v:.3f}" for p, v in zip(PCTS, q))
    print(f"{name:<16} n={len(sims):>7}  mean={sims.mean():.3f}  {line}")
    return q


def main():
    rng = np.random.default_rng(C.SEED)
    t_index = id_index(C.TERMS_CSV, "term_id")
    i_index = id_index(C.ITEMS_CSV, "item_id")
    term_emb = np.load(C.TERM_EMB_NPY)                      # 77 MB, RAM'e alinabilir
    item_emb = np.load(C.ITEM_EMB_NPY, mmap_mode="r")       # 1.48 GB, mmap

    train = pd.read_csv(C.TRAINING_PAIRS_CSV, dtype=str)
    sub = pd.read_csv(C.SUBMISSION_PAIRS_CSV, dtype=str)

    # test terimleri train'de var mi? (plan varsayimi: tamamen ayrik)
    tr_terms, te_terms = set(train["term_id"]), set(sub["term_id"])
    print(f"train terms={len(tr_terms)}  test terms={len(te_terms)}  kesisim={len(tr_terms & te_terms)}")
    per_term = sub.groupby("term_id").size()
    print(f"test cift/term: min={per_term.min()} median={per_term.median():.0f} "
          f"mean={per_term.mean():.1f} max={per_term.max()}")

    ts = sub.sample(min(N_SAMPLE, len(sub)), random_state=C.SEED)
    report("test_pairs", pair_sims(ts["term_id"].values, ts["item_id"].values,
                                   t_index, i_index, term_emb, item_emb))

    tp = train.sample(min(N_SAMPLE, len(train)), random_state=C.SEED)
    report("train_positives", pair_sims(tp["term_id"].values, tp["item_id"].values,
                                        t_index, i_index, term_emb, item_emb))

    all_items = np.array(list(i_index.keys()))
    rt = train["term_id"].sample(min(N_SAMPLE, len(train)), random_state=C.SEED + 1).values
    ri = rng.choice(all_items, size=len(rt))
    report("random_pairs", pair_sims(rt, ri, t_index, i_index, term_emb, item_emb))

    # test dagiliminin ham histogrami Faz 1'de mix ayari icin diske yazilir
    sims_test = pair_sims(ts["term_id"].values, ts["item_id"].values,
                          t_index, i_index, term_emb, item_emb)
    np.save(C.ARTIFACTS_DIR / "audit_test_sims.npy", sims_test)
    print(f"kaydedildi: {C.ARTIFACTS_DIR / 'audit_test_sims.npy'}")


if __name__ == "__main__":
    main()
