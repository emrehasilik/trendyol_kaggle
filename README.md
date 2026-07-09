# Trendyol Datathon 2

Kaggle yarışması: (arama terimi, ürün) çifti için 0/1 alaka tahmini, metrik macro-F1.
Veri: 250k pozitif eğitim çifti, 50.153 terim, 3.359.679 test çifti.

> Not: `data/`, `models/`, `output/`, `artifacts/` klasörleri boyut nedeniyle (~4.6GB, model
> dosyaları 350MB+) bu repoya dahil edilmedi. Burada sadece pipeline kodu bulunuyor.

> **Proje geçmişi, denenen tüm yöntemler ve güncel plan: [HANDOFF.md](HANDOFF.md).**

## v6 — Colab eğitimi (güncel aşama)

Büyük gövde (`BAAI/bge-reranker-v2-m3`) eğitimi Google Colab'da (A100/L4) yapılır:
[notebooks/colab_v6.ipynb](notebooks/colab_v6.ipynb) uçtan uca sürücüdür (Drive
checkpoint'li, oturum kopmasına dayanıklı). Kurulum adımları notebook'un başında ve
HANDOFF.md bölüm 7'de. İlgili scriptler: `src/tokenize_ce_cache_v6.py`,
`src/train_ce_v6.py`, `src/infer_ce_v6.py`, `src/build_dataset_v7.py`. Yerel
davranışı bozmadan ortam değişkenleriyle taşınabilirlik: `TK2_WORK`, `TK2_DATA`,
`TK2_MODEL`, `TK2_HF_CACHE` (bkz. `src/config.py`).

## Pipeline

1. **Negatif örnekleme** (`src/build_dataset.py`, `src/mine_ann.py`) — eğitim verisi sadece
   pozitif çiftler içeriyor, negatifler sentetik üretildi:
   - ANN top-200 komşuları (muhtemel gerçek pozitifler) negatif havuzundan tamamen yasaklandı.
   - Aday havuzu: yaprak kategori %50, üst kategori %25, rastgele %25.
   - Adaylar test setinin tahmini negatif-benzerlik histogramına oturtuldu.
   - Pozitif başına 4 negatif (~%20 pozitif oran).

2. **Cross-encoder eğitimi** (`src/train_ce.py`, `src/train_ce_v2.py`) —
   `dbmdz/bert-base-turkish-128k-uncased`, 3-fold GroupKFold (terim bazlı), 2 epoch, bf16.
   Girdi: `title | brand | son 2 kategori | gender/age | whitelist attribute`.

3. **LightGBM** (`src/train_lgbm.py`, `src/features.py`) — tablo tabanlı özelliklerle ek model.

4. **Stacker & blend** (`src/train_stacker.py`) — CE + LGBM çıktıları birleştirilip eşik optimize edildi.

5. **Inference** (`src/infer_ce.py`, `src/infer_ce_v2.py`, `src/predict_lgbm.py`) — test seti için
   submission dosyaları üretildi.

## Sonuçlar

- **v1:** CE OOF macro-F1 = 0.8995 (fold: 0.9000/0.8928/0.9055, eşik 0.43); LB-proxy F1 = 0.9210 (eşik 0.42).
- **Gerçek LB (v1 blend):** 0.822 — proxy ile ~0.10 kalibrasyon farkı (sentetik negatifler gerçek test
  negatiflerinden farklı davranıyor).
- **v2:** pseudo-label + CE-onaylı ANN hard negatifler ile genişletilmiş veri (1.18M satır, %29.5 pozitif).
  OOF (gerçek etiket) 0.9508, LB-proxy 0.9221 (eşik 0.73).

## Klasör yapısı (yerelde, repoya dahil değil)

```
data/       # ham veri (items, terms, training/submission pairs)
models/     # eğitilmiş CE/LGBM/stacker modelleri
output/     # submission CSV'leri
artifacts/  # OOF skorları, embedding/ANN cache, log dosyaları
```
