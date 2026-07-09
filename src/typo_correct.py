"""Sorgu yazim duzeltmesi: katalog kelime dagarcigina gore typo onarimi.

Gri bolge incelemesi tipik hatalari gosterdi: 'leptop'->laptop,
'shefer'->schafer, 'sehba'->sehpa. Dusuk skorun sebebi urunun alakasizligi
degil sorgunun bozuklugu olabilir.

Yontem: item title+brand+category tokenlarindan frekansli sozluk; sorgu
tokeni sozlukte yoksa ayni bas harf + benzer uzunluktaki adaylar icinde
difflib oraniyla en iyi eslesme (frekans agirlikli). Muhafazakar esikler.

Cikti: artifacts/terms_corrected.csv (term_id, query, corrected, changed)
"""
import difflib
import re
from collections import Counter, defaultdict

import pandas as pd

import config as C

TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")
MIN_VOCAB_FREQ = 10     # sozlukte "gecerli kelime" sayilmak icin
MIN_TOKEN_LEN = 4       # bundan kisa tokenlara dokunma
SIM_CUT = 0.78          # difflib benzerlik alt esigi


def tokens(s):
    return TOKEN_RE.findall(s.lower())


def main():
    # ---- katalog sozlugu
    vocab = Counter()
    reader = pd.read_csv(C.ITEMS_CSV, dtype=str, keep_default_na=False,
                         usecols=["title", "brand", "category"], chunksize=100_000)
    for chunk in reader:
        for col in ("title", "brand", "category"):
            for text in chunk[col].values:
                vocab.update(tokens(text))
    good = {w for w, c in vocab.items() if c >= MIN_VOCAB_FREQ}
    print(f"sozluk: {len(vocab)} token, frekans>={MIN_VOCAB_FREQ}: {len(good)}")

    # aday indeksi: (bas harf, uzunluk) -> kelime listesi (frekansa gore sirali)
    index = defaultdict(list)
    for w in sorted(good, key=lambda x: -vocab[x]):
        index[(w[0], len(w))].append(w)

    def correct_token(t):
        if len(t) < MIN_TOKEN_LEN or t in good or t.isdigit():
            return t
        cands = []
        for dl in (0, 1, -1, 2):
            cands.extend(index.get((t[0], len(t) + dl), [])[:4000])
        if not cands:
            return t
        best = difflib.get_close_matches(t, cands, n=1, cutoff=SIM_CUT)
        return best[0] if best else t

    terms = pd.read_csv(C.TERMS_CSV, dtype=str, keep_default_na=False)
    terms["query"] = terms["query"].str.lower()

    # sorgu-frekans korumasi: >=3 farkli sorguda gecen token typo sayilmaz
    # ('lanolin', 'siamp', '16pro' gibi katalog-disi ama gecerli kelimeler)
    qvocab = Counter()
    for q in terms["query"].values:
        qvocab.update(set(tokens(q)))
    good |= {w for w, c in qvocab.items() if c >= 3}
    print(f"sorgu-frekans korumasiyla sozluk: {len(good)}")

    cache = {}
    corrected = []
    for q in terms["query"].values:
        toks = tokens(q)
        out = []
        for t in toks:
            if t not in cache:
                cache[t] = correct_token(t)
            out.append(cache[t])
        corrected.append(" ".join(out))
    terms["corrected"] = corrected
    # tokenlari birlestirilmis normal hali de degisiklik sayilmasin
    norm = terms["query"].map(lambda q: " ".join(tokens(q)))
    terms["changed"] = terms["corrected"] != norm

    print(f"duzeltilen sorgu: {terms['changed'].sum()} / {len(terms)} "
          f"(%{100 * terms['changed'].mean():.1f})")
    ex = terms[terms["changed"]].sample(15, random_state=3)
    for _, r in ex.iterrows():
        print(f"  '{r['query']}' -> '{r['corrected']}'")
    terms.to_csv(C.ARTIFACTS_DIR / "terms_corrected.csv", index=False)
    print("kaydedildi: terms_corrected.csv")


if __name__ == "__main__":
    main()
