"""Coklu dil destekli embedding modeli ile anlamsal benzerlik.

TF-IDF kelime bazli oldugu icin es anlamli/varyant kelimeleri (ornek:
"ayakkabi" vs "sneaker", ya da farkli ek almis kelimeler) yakalayamiyor.
Bu modul query ve title'lari anlam uzayinda vektore cevirip cosine
similarity hesabini saglar - TF-IDF'in kor kaldigi durumlari telafi eder.

Onemli: embedding'ler SADECE egitim (fit) asamasinda, tum katalog (terms.csv +
items.csv) icin bir kere hesaplanir ve diske kaydedilir. Tahmin (predict)
asamasinda model bir daha YUKLENMEZ - sadece kaydedilmis sayisal vektorler
okunur, bu yuzden internet gerektirmez.
"""
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_FOLDER = "models/st_cache"


def get_model(device: str = None):
    from sentence_transformers import SentenceTransformer
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, cache_folder=CACHE_FOLDER, device=device)


def encode_texts(model, texts, batch_size: int = 256, chunk_size: int = 50_000) -> np.ndarray:
    """Buyuk listeleri parca parca encode eder.

    sentence-transformers tum sonuclari tek seferde bellekte topluyor; 1M'e
    yakin metinde bu, gecici bellek tepe noktasini ikiye katlayip OOM'a
    sebep olabiliyor. Disardan chunk'layip dogrudan onceden ayrilmis bir
    float32 diziye yazmak tepe bellek kullanimini dusuruyor.
    """
    texts = list(texts)
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    out = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        emb = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize -> cosine sim = dot product
        )
        out[start:end] = emb.astype(np.float32)
    return out
