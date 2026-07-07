"""RAM-yalin feature pipeline'i (kompakt token artifact'i uzerinden).

Once prepare_item_tokens.py calistirilmis olmali (artifacts/item_tokens.npz).
Eski TF-IDF cosine yerine iki ucuz leksik sinyal: q_coverage ve jaccard.
Peak RAM < 1 GB hedeflenir; embeddingler mmap.
"""
import re
import zlib

import numpy as np
import pandas as pd

import config as C
from data_utils import fold_tr

TOKEN_RE = re.compile(r"[a-z0-9]+")

FEATURE_COLS = [
    "embed_sim", "q_coverage", "jaccard", "gender_match", "age_match",
    "brand_in_query", "category_overlap", "attr_overlap",
]

# item_tokens.npz'deki kodlarla ayni (prepare_item_tokens.py)
_G_KW = {1: {"kadin", "bayan"}, 2: {"erkek"}, 3: {"unisex"}}
_A_KW = {1: {"cocuk"}, 2: {"bebek"}, 3: {"yetiskin"}, 4: {"genc"}}


def _hash_set(text):
    return {zlib.crc32(t.encode()) for t in TOKEN_RE.findall(text)} if text else set()


class LeanFeatures:
    def __init__(self):
        tk = np.load(C.ARTIFACTS_DIR / "item_tokens.npz")
        self.tk = {k: tk[k] for k in tk.files}

        terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
        self.t_index = {v: i for i, v in enumerate(terms["term_id"].values)}
        item_ids = pd.read_csv(C.ITEMS_CSV, usecols=["item_id"], dtype=str)["item_id"].values
        self.i_index = {v: i for i, v in enumerate(item_ids)}

        # sorgu tarafi: 50k term icin hash seti + kelime seti (gender/age kw icin)
        self.q_hash, self.q_words = {}, {}
        for tid, q in zip(terms["term_id"], terms["query"]):
            qf = fold_tr(q.lower())
            words = set(TOKEN_RE.findall(qf))
            self.q_words[tid] = words
            self.q_hash[tid] = {zlib.crc32(w.encode()) for w in words}

        self.term_emb = np.load(C.TERM_EMB_NPY)
        self.item_emb = np.load(C.ITEM_EMB_NPY, mmap_mode="r")

    def _slice(self, field, i):
        off = self.tk[f"{field}_off"]
        return self.tk[f"{field}_vals"][off[i]:off[i + 1]]

    def transform(self, pairs_df, chunk_size=200_000):
        n = len(pairs_df)
        out = {c: np.zeros(n, dtype=np.float32) for c in FEATURE_COLS}
        term_ids = pairs_df["term_id"].values
        item_ids = pairs_df["item_id"].values

        i_rows_all = np.fromiter((self.i_index[i] for i in item_ids),
                                 dtype=np.int64, count=n)
        t_rows_all = np.fromiter((self.t_index[t] for t in term_ids),
                                 dtype=np.int64, count=n)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            out["embed_sim"][s:e] = np.sum(
                self.term_emb[t_rows_all[s:e]] * np.asarray(self.item_emb[i_rows_all[s:e]]),
                axis=1)

        gender = self.tk["gender"]
        age = self.tk["age"]
        t_off, t_vals = self.tk["title_off"], self.tk["title_vals"]
        b_off, b_vals = self.tk["brand_off"], self.tk["brand_vals"]
        c_off, c_vals = self.tk["cat_off"], self.tk["cat_vals"]
        a_off, a_vals = self.tk["attr_off"], self.tk["attr_vals"]

        for g in range(n):
            tid = term_ids[g]
            q_h = self.q_hash.get(tid)
            if not q_h:
                continue
            q_w = self.q_words[tid]
            i = i_rows_all[g]

            tt = t_vals[t_off[i]:t_off[i + 1]]
            inter = 0
            for h in tt:
                if h in q_h:
                    inter += 1
            out["q_coverage"][g] = inter / len(q_h)
            union = len(q_h) + len(tt) - inter
            out["jaccard"][g] = inter / union if union else 0.0

            gc = gender[i]
            if gc and (q_w & _G_KW[gc]):
                out["gender_match"][g] = 1
            ac = age[i]
            if ac and (q_w & _A_KW[ac]):
                out["age_match"][g] = 1

            bb = b_vals[b_off[i]:b_off[i + 1]]
            for h in bb:
                if h in q_h:
                    out["brand_in_query"][g] = 1
                    break

            cc = c_vals[c_off[i]:c_off[i + 1]]
            if len(cc):
                ci = 0
                for h in cc:
                    if h in q_h:
                        ci += 1
                out["category_overlap"][g] = ci / len(cc)

            aa = a_vals[a_off[i]:a_off[i + 1]]
            if len(aa):
                ai = 0
                for h in aa:
                    if h in q_h:
                        ai += 1
                out["attr_overlap"][g] = ai / len(q_h)

        return pd.DataFrame(out)
