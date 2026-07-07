"""Feature engineering: TF-IDF cosine similarity + kategorik eslesme flag'leri.

Strateji 2: query/title TF-IDF -> cosine similarity + kategori/marka/cinsiyet
eslesme flag'leri -> LightGBM'e besleniyor.
"""
import re
import gc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import parse_attributes
import embeddings as emb_module

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Not: data_utils.fold_tr ile metin ASCII'ye indirgendigi icin burada da
# katlanmis (sş->s, ç->c, ı->i vb.) formlar kullaniliyor.
GENDER_KEYWORDS = {
    "kadin": {"kadin", "bayan"},
    "erkek": {"erkek"},
    "unisex": {"unisex"},
}
AGE_KEYWORDS = {
    "cocuk": {"cocuk"},
    "bebek": {"bebek"},
    "yetiskin": {"yetiskin"},
    "genc": {"genc"},
}


def tokenize(text: str) -> set:
    if not text:
        return set()
    return set(TOKEN_RE.findall(text.lower()))


class FeatureBuilder:
    def __init__(self, max_features: int = 60_000, char_max_features: int = 40_000):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        # Turkce eklemeli (agglutinative) bir dil: "ayakkabi" (query) ile
        # "ayakkabisi" (title) kelime bazinda FARKLI token'dir. Karakter
        # n-gram'lari (3-5 harf) sonek farkini tolere eder, kelime kokunu yakalar.
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=char_max_features,
            sublinear_tf=True,
        )
        self.term_vec = None      # sparse matrix, normalized, indexed by term_id order
        self.item_vec = None      # sparse matrix, normalized, indexed by item_id order
        self.term_char_vec = None
        self.item_char_vec = None
        self.term_index = {}      # term_id -> row idx
        self.item_index = {}      # item_id -> row idx
        self.item_category_tokens = {}
        self.item_brand_tokens = {}
        self.item_gender = {}
        self.item_age_group = {}
        self.item_attr_values = {}   # item_id -> set of attribute value tokens
        self.term_emb = None      # dense float32, L2-normalized, indexed gibi term_vec
        self.item_emb = None      # dense float32, L2-normalized, indexed gibi item_vec
        self.use_embeddings = False

    def fit(self, terms_df: pd.DataFrame, items_df: pd.DataFrame, use_embeddings: bool = True, embed_batch_size: int = 256):
        corpus = list(terms_df["query"]) + list(items_df["title"])
        self.vectorizer.fit(corpus)

        term_mat = self.vectorizer.transform(terms_df["query"])
        item_mat = self.vectorizer.transform(items_df["title"])
        self.term_vec = _l2_normalize(term_mat).astype(np.float32)
        self.item_vec = _l2_normalize(item_mat).astype(np.float32)
        del term_mat, item_mat

        self.char_vectorizer.fit(corpus)
        del corpus
        term_char_mat = self.char_vectorizer.transform(terms_df["query"])
        item_char_mat = self.char_vectorizer.transform(items_df["title"])
        self.term_char_vec = _l2_normalize(term_char_mat).astype(np.float32)
        self.item_char_vec = _l2_normalize(item_char_mat).astype(np.float32)
        del term_char_mat, item_char_mat
        gc.collect()

        self.term_index = {tid: i for i, tid in enumerate(terms_df["term_id"].values)}
        self.item_index = {iid: i for i, iid in enumerate(items_df["item_id"].values)}

        self.use_embeddings = use_embeddings
        if use_embeddings:
            # Embedding modeli dogal Turkce metin uzerinde egitildi - TF-IDF icin
            # ASCII'ye katlanmis (fold_tr) metin yerine orijinal (sadece lowercase)
            # query_raw/title_raw kullaniyoruz, yoksa semantik kalite dusuyor.
            model = emb_module.get_model()
            query_col = terms_df["query_raw"] if "query_raw" in terms_df.columns else terms_df["query"]
            title_col = items_df["title_raw"] if "title_raw" in items_df.columns else items_df["title"]
            self.term_emb = emb_module.encode_texts(model, query_col, batch_size=embed_batch_size)
            self.item_emb = emb_module.encode_texts(model, title_col, batch_size=embed_batch_size)
            del model
            gc.collect()

        for row in items_df.itertuples(index=False):
            iid = row.item_id
            self.item_category_tokens[iid] = tokenize(row.category)
            self.item_brand_tokens[iid] = tokenize(row.brand)
            self.item_gender[iid] = row.gender
            self.item_age_group[iid] = row.age_group
            attrs = parse_attributes(row.attributes)
            val_tokens = set()
            for v in attrs.values():
                val_tokens |= tokenize(v)
            self.item_attr_values[iid] = val_tokens

        return self

    def transform(self, pairs_df: pd.DataFrame, terms_df: pd.DataFrame, chunk_size: int = 200_000) -> pd.DataFrame:
        query_lookup = dict(zip(terms_df["term_id"], terms_df["query"]))

        n = len(pairs_df)
        out_cosine = np.zeros(n, dtype=np.float32)
        out_char_cosine = np.zeros(n, dtype=np.float32)
        out_embed_sim = np.zeros(n, dtype=np.float32)
        out_gender_match = np.zeros(n, dtype=np.int8)
        out_age_match = np.zeros(n, dtype=np.int8)
        out_brand_in_query = np.zeros(n, dtype=np.int8)
        out_category_overlap = np.zeros(n, dtype=np.float32)
        out_attr_overlap = np.zeros(n, dtype=np.float32)

        term_ids = pairs_df["term_id"].values
        item_ids = pairs_df["item_id"].values

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            t_idx = np.array([self.term_index.get(t, -1) for t in term_ids[start:end]])
            i_idx = np.array([self.item_index.get(i, -1) for i in item_ids[start:end]])

            valid = (t_idx >= 0) & (i_idx >= 0)
            cos_chunk = np.zeros(end - start, dtype=np.float32)
            if valid.any():
                q_sub = self.term_vec[t_idx[valid]]
                d_sub = self.item_vec[i_idx[valid]]
                cos_chunk[valid] = np.asarray(q_sub.multiply(d_sub).sum(axis=1)).ravel()
            out_cosine[start:end] = cos_chunk

            char_cos_chunk = np.zeros(end - start, dtype=np.float32)
            if valid.any():
                qc_sub = self.term_char_vec[t_idx[valid]]
                dc_sub = self.item_char_vec[i_idx[valid]]
                char_cos_chunk[valid] = np.asarray(qc_sub.multiply(dc_sub).sum(axis=1)).ravel()
            out_char_cosine[start:end] = char_cos_chunk

            if self.use_embeddings and self.term_emb is not None:
                embed_chunk = np.zeros(end - start, dtype=np.float32)
                if valid.any():
                    qe = self.term_emb[t_idx[valid]]
                    de = self.item_emb[i_idx[valid]]
                    embed_chunk[valid] = np.sum(qe * de, axis=1)
                out_embed_sim[start:end] = embed_chunk

            for local_pos, global_pos in enumerate(range(start, end)):
                t_id = term_ids[global_pos]
                i_id = item_ids[global_pos]
                query = query_lookup.get(t_id, "")
                q_tokens = tokenize(query)

                gender = self.item_gender.get(i_id, "unknown")
                if gender in GENDER_KEYWORDS and (q_tokens & GENDER_KEYWORDS[gender]):
                    out_gender_match[global_pos] = 1

                age = self.item_age_group.get(i_id, "unknown")
                if age in AGE_KEYWORDS and (q_tokens & AGE_KEYWORDS[age]):
                    out_age_match[global_pos] = 1

                brand_tokens = self.item_brand_tokens.get(i_id, set())
                if brand_tokens and (brand_tokens & q_tokens):
                    out_brand_in_query[global_pos] = 1

                cat_tokens = self.item_category_tokens.get(i_id, set())
                if cat_tokens and q_tokens:
                    overlap = len(cat_tokens & q_tokens) / len(cat_tokens)
                    out_category_overlap[global_pos] = overlap

                attr_tokens = self.item_attr_values.get(i_id, set())
                if attr_tokens and q_tokens:
                    a_overlap = len(attr_tokens & q_tokens) / max(len(q_tokens), 1)
                    out_attr_overlap[global_pos] = a_overlap

        result = {
            "cosine_sim": out_cosine,
            "char_cosine_sim": out_char_cosine,
            "gender_match": out_gender_match,
            "age_match": out_age_match,
            "brand_in_query": out_brand_in_query,
            "category_overlap": out_category_overlap,
            "attr_overlap": out_attr_overlap,
        }
        if self.use_embeddings:
            result["embed_sim"] = out_embed_sim
        return pd.DataFrame(result)

    def save(self, path: str):
        # Embedding matrisleri buyuk (yuzlerce MB - GB), ana pickle'i sismemesin
        # diye ayri .npy dosyalarinda saklaniyor.
        term_emb, item_emb = self.term_emb, self.item_emb
        self.term_emb, self.item_emb = None, None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.term_emb, self.item_emb = term_emb, item_emb
        if term_emb is not None:
            np.save(path + ".term_emb.npy", term_emb)
            np.save(path + ".item_emb.npy", item_emb)

    @staticmethod
    def load(path: str) -> "FeatureBuilder":
        with open(path, "rb") as f:
            fb = pickle.load(f)
        if fb.use_embeddings:
            fb.term_emb = np.load(path + ".term_emb.npy")
            fb.item_emb = np.load(path + ".item_emb.npy")
        return fb


def _l2_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(mat.multiply(mat).sum(axis=1))
    norms = np.asarray(norms).ravel()
    norms[norms == 0] = 1.0
    inv = sp.diags(1.0 / norms)
    return (inv @ mat).tocsr()
