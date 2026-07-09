# TRENDYOL DATATHON 2 — PROJE DEVIR BELGESI (HANDOFF)

> Bu belge yeni bir Claude Code oturumuna projeyi devretmek icin yazildi.
> Durum tarihi: 2026-07-09. En iyi LB: **0.851** (siramiz ~132; ilk 10 ~0.92).
> Yeni oturumun gorevi en altta: "YENI OTURUMUN GOREVI" bolumu.

---

## 1. YARISMA

- **Gorev:** (arama terimi, urun) cifti icin binary alaka tahmini (1=relevant, 0=irrelevant).
- **Metrik:** macro-F1 (iki sinifin F1 ortalamasi — iki sinifta da iyi olmak sart).
- **Tuzak:** egitim verisi YALNIZCA pozitif cift icerir; negatif uretimi tamamen bize kalmis.
- **Degerlendirme:** Public LB canli; siralama PRIVATE subset ile. Private top-10 finale
  (fiziksel hackathon) kalir. Kaggle skoru toplam puanin %40'i; hackathon (hiz,
  aciklanabilirlik arayuzu, sunum, rapor) %60'i.
- **Kurallar (kritik olanlar):** gunde 5 submission; yarisma sonunda EN FAZLA 2 submission
  secilir (Kaggle'da "Select Submission" unutulmasin — su an secili olmasi gereken:
  `sub_v3_ce_rate25.csv`); ilk 20 takimdan YENIDEN URETILEBILIR kod istenecek
  (versiyonlu script yapimiz buna hazir); veri seti ucuncu taraflarla paylasilamaz
  (harici LLM API'ye veri gondermek YASAK — bunu degerlendirdik ve eledik);
  takimlar arasi transfer yasak.

## 2. VERI

| Dosya | Icerik |
|---|---|
| training_pairs.csv | 250.000 POZITIF (term_id, item_id) cifti; 17.968 benzersiz terim |
| terms.csv | 50.153 terim (query metni). Train ve test terimleri HIC KESISMEZ |
| items.csv | 962.873 urun: title, category (hiyerarsik, '/' ayracli), brand, gender, age_group, attributes ("k: v, k: v" formatinda) |
| submission_pairs.csv | 3.359.679 test cifti = 32.185 terim x ~100 aday (min 100, medyan 100) |

- Test listeleri Trendyol'un aday uretiminden geliyor; etiketler buyuk olasilikla
  gercek kullanici davranisindan (typo'lu sorgu yazan kullanici dogru urune tiklamis
  → typo duzeltmesi ground-truth ile HIZALI).
- CE girdi sablonu (`src/text_builder.py`): `title | brand | son 2 kategori segmenti |
  gender age | whitelist attribute'lar (13 anahtar, max 8 k:v)` → ort. 36 BERTurk tokeni.

## 3. ITERASYONLAR — YONTEM, SONUC, DERS

### v1 — Temel pipeline → **LB 0.822**
**Negatif uretimi** (`src/build_dataset.py`, kalbi bu):
- Eski projeden 384-dim bi-encoder embeddingleri (config.py'de path'ler). Her terimin
  ANN top-200 komsusu cikarildi (`mine_ann.py`). Audit: bu bant (sim 0.75-0.90) buyuk
  olasilikla ETIKETLENMEMIS GERCEK POZITIF → negatif havuzundan TAMAMEN YASAK + sim<0.78 tavan.
- Aday havuzlari: pozitifin yaprak kategorisi %50, ust kategori %25, rastgele %25 (2.5x oversample).
- **Histogram esleme:** aday sim'leri, testin tahmini negatif-sim dagilimina
  (test_hist − 0.22·pos_hist) bin bazli agirlikli orneklemeyle oturtuldu.
- Pozitif basina 4 negatif → ~950K satir (%20 poz).
- **LB-proxy:** 2000 terim tamamen ayrildi, test benzeri ~100'luk listeler (validasyon).
**Modeller:** LGBM (el yapimi ozellikler; proxy 0.866) · **CE: dbmdz/bert-base-turkish-128k-uncased**,
3-fold GroupKFold(term), 2 epoch, lr 2e-5, bs 32 x accum 2, bf16, maxlen 128 (proxy 0.921)
· stacker (0.906, cop) · **blend 0.7·CE + 0.3·LGBM (proxy 0.9219)**.
**Sonuc:** LB 0.822. **DERS #1:** proxy 0.9219 vs LB 0.822 → sentetik negatifler gercek
test negatiflerine benzemiyor; proxy'nin cevap anahtari bizim uydurmamiz.

### v2 — Pseudo-label + hard negatif → **LB 0.845 (+0.023, en buyuk sicrama)**
1. **Pseudo-labeling:** blend'in test uzerindeki emin tahminleri egitime girdi
   (>0.95 → 128K pozitif, <0.05 → 403K negatif; terim basina tavan 15/13).
   Train-test terimleri ayrik oldugundan model ilk kez GERCEK test dagilimini gordu.
2. **CE-onayli hard negatifler:** v1'de yasakli ANN bandini v1 CE puanladi
   (`vet_ann_negatives.py`); skor<0.30 olan 250K "cok benzer ama alakasiz" cift eklendi.
3. Çapa: 221K gercek pozitif + 180K v1 negatifi. Toplam 1.18M satir, %29.5 poz.
   Mimari/hiperparametre AYNI (kontrollu deney).
**Sonuc:** LB 0.845. **DERS #2:** dagilim duzeltmeleri proxy'de gorunmez (proxy +0.001
demisti, gercek +0.023) → submission haklari deney/olcum araci olarak kullanilmali.

### v3 — Dongu 2. turu + ESIK KALIBRASYONU → **LB 0.843; @%25 oran = 0.851 (EN IYI)**
- Ayni tarif, ogretmen v2: pl_pos 306K, pl_neg 469K, annvet v2'yle yeniden vetlendi
  (<0.25), v1neg 120K → 1.37M satir.
- LB 0.843 → **DERS #3: self-training 1 tur calisir, 2. turda doyar** (ogretmen kendi
  inanclarini aktarir; hatalar pseudo-etiketin ulasamadigi gri bolgede).
- **ORAN EGRISI (altin bulgu):** ayni v3 skorlari, farkli pozitif oranlarla kesildi:
  %22.7→0.843, **%25.0→0.851**, %28→0.839. **DERS #4: TEPE %25'TE.** Proxy-sweep esikleri
  sistematik dusuk oran seciyordu. ARTIK HER SUBMISSION %25 pozitif oranla kesilir:
  `thr = np.quantile(scores, 0.75)`.
- **EN IYI DOSYA: `output/sub_v3_ce_rate25.csv`** = v3'un 3 fold ortalama test skoru
  (`artifacts/ce_v3_test_mean.npy`), %25 oranla binarize.

### v4 — Mimari cesitliligi denemesi → NOTR, aranan yontem BU DEGILDI
(Karistirma: v4 istedigimiz sicramayi VERMEDI; sadece kayit icin.)
- Amac ensemble cesitliligiydi. mDeBERTa 6GB laptopta iki kez basarisiz (NaN + bellek
  tasip Windows paylasimli RAM'ine sizma = sessiz 10-20x yavaslama). XLM-R-base'e gecildi
  (embedding dondurma hilesiyle sigdi). Uzlasma-filtreli pseudo-etiketler uretildi
  (`train_dataset_v4.parquet`: v2 VE v3 hemfikirse etiket — EN TEMIZ VERIMIZ, 1.36M satir).
- Sonuc: v2+v3+v4 ensemble @%25 → LB 0.850 ≈ notr. **DERS #5: ayni pseudo-etiket
  soyundan beslenen modeller gri bolgede AYNI hatalari yapar; yeni MIMARI degil yeni
  BILGI gerekir.** (Not: zayif v4'un sebebi 6GB kisitlamalariydi — buyuk GPU'da
  guclu gövde + ayni veri hala en umutlu yol.)

### Olculmus OLU YOLLAR (tekrar denenmesin)
- Cinsiyet/yas celiski kurallari: model zaten biliyor (40.351 celiskili test ciftinin
  sadece 170'i pozitif tahminli; kural guvenilirligi gercek pozitiflerde %99.8 ama duzeltilecek bir sey yok).
- Liste-ici kategori oylamasi: gri bolgede P(destek|y=1)=0.94 vs P(destek|y=0)=0.91 — ayirt etmiyor.
- Liste-ici rank/z-skor esikleri: hepsi global esikten kotu.
- %25 disinda oran: %28 → 0.839 (sert dusus).

### CANLI ipuclari
- **Typo duzeltme:** sorgularin %3.4'u typo'lu; bu terimler proxy'de 0.72 F1 (ortalama 0.92!).
  `src/typo_correct.py` (katalog+sorgu-frekans sozluklu difflib duzeltici) +
  `src/rescore_typofix.py` (v3 ile rescore, max-blend). Gercek pozitif ort skor
  0.589→0.827. Dosya: `sub_ens_v234_typofix_rate25.csv` (LB sonucu henuz bilinmiyor).
- **Gri bolge:** 320K cift (%9.5, skor 0.05-0.95). Icerik: ince urun-tipi anlami
  ("patik" vs "makosen"), marka bilgisi ("leptop monster" → Monster'in Abra modeli), typolar.
- Skor gecmisi ozeti: v1 0.822 → v2 0.845 → v3@25 **0.851** → ens 0.850 (notr) → sira ~132, top10 ~0.92.

## 4. REPO / DOSYA HARITASI

- `src/`: tum pipeline (config.py'de tum path/hiperparametreler). Onemli akis:
  `mine_ann.py → build_dataset.py → tokenize_ce_cache.py → train_ce.py → infer_ce.py`
  ve iterasyonlar: `vet_ann_negatives.py → build_dataset_v{2,3,4,5}.py →
  train_ce_v{2,3}.py → infer_ce_v{2,3}.py`. Ilerleme: `progress.py [v2|v3|...]`.
- `artifacts/`: skorlar (`ce_v3_test_mean.npy`, `ens_v234_test.npy`...), datasetler
  (`train_dataset_v4.parquet` = en temiz), token cache'leri (tokenizer'a OZEL —
  yeni model = yeni cache), loglar.
- `models/`: `ce_fold*.pt` (v1), `ce_v2_*`, `ce_v3_*` (fp16 state_dict).
- `output/`: tum submissionlar. EN IYI: `sub_v3_ce_rate25.csv` (0.851).
- GitHub: https://github.com/emrehasilik/trendyol_kaggle (sadece src+README;
  data/models/artifacts hariç — boyut). Bu handoff da repoya eklenecek.
- Eski proje bagimliligi: bi-encoder embeddingler
  `C:\Users\Hp\Desktop\project\trendyol_kaggle\models\feature_builder.pkl.*_emb.npy`
  (Colab'da YOK — ANN/negatif uretimi yeniden gerekirse embeddingler de uretilmeli
  ya da mevcut parquet'ler tasinmali).

## 5. DONANIM DERSLERI (yerel laptop: RTX 3060 6GB)
- 6GB'ta buyuk modeller: mDeBERTa imkansiz (disentangled attention bmm patlamasi);
  XLM-R ancak embedding dondurup GC ile. Windows WDDM bellek tasmasinda HATA VERMEZ,
  sessizce sistem RAM'ine tasar → 10-20x yavaslama; belirti "GPU %100 ama log ilerlemiyor".
- Bu yuzden buyuk model egitimi COLAB'a tasiniyor (A100/L4/T4). T4'te bf16 YOK
  (fp16+GradScaler gerekir; mDeBERTa fp16'da NaN'a meyilli → T4'te mdeberta'dan kacin);
  L4/A100'de bf16 var, dert yok.
- Her uzun kosudan once 50 adimlik olcum (hiz+VRAM+NaN), kosuda log-tempo gozculugu sart.

---

## 6. YENI OTURUMUN GOREVI (TAMAMLANDI — plan ve kod bolum 7'de)

**Hedef: 0.851 → 0.92+ (private top-10 icin gereken seviye). Yontem karari yeni
oturumundur; asagidaki cerceve onerilir:**

1. Kullanici projeyi GitHub'a pushlayacak; **Google Colab** (yuksek VRAM GPU) uzerinde
   repo'yu cekip egitim yapilacak. Yeni oturumun ilk isi: Colab notebook'u hazirlamak
   (repo clone + Kaggle API ile veri indirme + egitim + test skorlama + %25 oranli
   submission uretimi; Drive'a checkpoint kaydi; 12 saatlik oturum kopmalarina karsi
   resumable).
2. **Model sinifi atlamasi (ana hamle):** BERTurk-base yerine guclu cok-dilli reranker
   gövdesi. Guclu adaylar: `BAAI/bge-reranker-v2-m3` (XLM-R-large gövdeli, dogrudan
   sorgu-urun alaka icin on-egitimli — en guclu aday), `microsoft/mdeberta-v3-base`
   (bf16 destekli GPU'da sorunsuz), `xlm-roberta-large`. Girdiyi zenginlestir
   (maxlen 192-256, attribute whitelist'i genislet).
3. **Veri:** `train_dataset_v4.parquet` (uzlasma-filtreli pseudo-etiketler) en temiz
   baslangic. Buyuk model egitilince PSEUDO-LABEL DONGUSUNU YENI MODELLE BIR TUR DAHA
   cevir (bu kez ogretmen farkli sinif → gercekten yeni bilgi; DERS #3'e takilmaz).
4. **Sabitler:** esik daima %25 pozitif oran (`np.quantile(s, 0.75)`); her submission
   tek degisken test etsin; proxy'ye kucuk farklar icin GUVENME (DERS #2); final 2
   submission secimini guncel tut.
5. Zaman: yarisma bitimine ~10 gun. Hackathon hazirligi (%60 puan: hiz, aciklanabilirlik,
   sunum) icin de son gunlerde pay ayrilmali.

---

## 7. v6 PLANI — MODEL SINIFI ATLAMASI (Colab, 2026-07-09'da kuruldu)

### Karar ve gerekce
- **Govde: `BAAI/bge-reranker-v2-m3`** (XLM-R-large, 568M param). Secim gerekcesi:
  dogrudan sorgu-pasaj alakasi icin on-egitimli reranker — bizim gorevin TA KENDISI;
  `num_labels=1` basligi hazir geliyor (sicak baslangic, pipeline'imizla birebir uyumlu:
  BCE + sigmoid). Cok dilli (Turkce dahil). Yedek: `microsoft/mdeberta-v3-base`
  (`TK2_MODEL` env ile tek satirda degisir; YALNIZ bf16 GPU'da — fp16'da NaN, bkz. bolum 5).
- **Veri: `train_dataset_v5.parquet`** (1.4M satir; v4 uzlasma etiketleri + proxy'nin
  ~29K GERCEK pozitifi + dengeleyici proxy negatifleri). Lokalde uretildi, hic egitilmedi
  (BERTurk'le v5 egitimi fold0/adim500'de birakildi — buyuk govde beklemeye alindi).
- **TEK model, TUM veri, 3-fold YOK.** Gerekce: proxy pusulaliktan emekli; OOF cogunlukla
  pseudo uzerinde oldugundan metrik degil; fold ayirmak %33 veri kaybi + 3x inference.
  Izleme icin %2 terim holdout (yalniz NaN/raydan-cikma gozculugu, model secimi degil).
- **Girdi zenginlestirme:** maxlen 128→192, item payi 110→160 token, tam kategori yolu
  (eskiden son 2 segment), attribute tavani 8→10. Metin ORIJINAL harf durumuyla
  (XLM-R cased; BERTurk-uncased'in lowercase kurali v6'da gecersiz).
- **v7 = pseudo-label dongusu YENI ogretmenle:** v6 farkli model sinifi oldugundan
  gri-bolge hatalari farkli → dongu gercekten yeni bilgi tasir (DERS #3/#5'e takilmaz).
  Eski v2/v3-soylu pseudo'lar ATILIR; capalar (pos/annvet/v1neg/proxy_*) korunur,
  pseudo'lar `ce_v6_test.npy`den yeniden uretilir (esikler 0.95/0.05, tavan 18/15 — v4 tarifi).

### Yeni dosyalar
- `src/tokenize_ce_cache_v6.py` — yeni tokenizer cache'i (dosya adinda model slug'i var,
  cache karismasi imkansiz).
- `src/train_ce_v6.py` — egitim. Adim-bazli resumable (20 dk'da bir model fp16 +
  global_step Drive'a atomik yazilir; optimizer state kaydedilmez, resume'da 200 adim
  lr yeniden-isinmasi). bf16/fp16 ve batch otomatik (A100 64x1, L4 32x2, T4 16x4 —
  efektif hep 64). Ilk 50 adimda hiz/VRAM/loss olcumu; 50 ardisik NaN'da durur.
  `--data ... --tag ...` ile v7'de aynen yeniden kullanilir.
- `src/infer_ce_v6.py` — 3.36M cifti 262K'lik chunk'larla skorlar (chunk'lar Drive'a
  yazilir, resume'da atlanir); `%25` oranla submission uretir.
- `src/build_dataset_v7.py` — v6 ogretmenli pseudo dongusu dataseti.
- `notebooks/colab_v6.ipynb` — uctan uca surucu: Drive mount + repo clone + Kaggle API
  veri + cache + egitim + skorlama + submission + v7 dongusu + blend hucresi.
- `src/config.py` — env override'lar: `TK2_WORK` (artifacts/models/output → Drive),
  `TK2_DATA`, `TK2_HF_CACHE`, `TK2_MODEL`, `TK2_CKPT_MIN`. Env yoksa lokal davranis AYNI.

### Colab is akisi (tek seferlik kurulum)
1. Drive'da `MyDrive/trendyol_v6/` olustur: icine `kaggle.json` +
   `artifacts/train_dataset_v5.parquet` (22MB, laptoptan) + istege bagli
   `artifacts/proxy_lists.parquet` (sanity) ve `artifacts/ce_v3_test_mean.npy` (blend).
2. Repoyu GitHub'a pushla (yalniz kod; parquet'ler REPOYA KONMAZ — kural: veri
   ucuncu tarafla paylasilamaz, public repoya veri turevleri koymak riskli).
3. Notebook'u Colab'da ac (A100 sec), `COMPETITION` slug'ini doldur, hucreleri sirayla
   calistir. Oturum koparsa: yeniden baglan, hepsini bastan calistir (her adim resumable).
4. Sure: A100 ~3-4 saat egitim + ~30 dk skorlama; L4 ~8-10 saat + ~1.5 saat.

### Submission sirasi (her biri tek degisken, hepsi %25)
1. `sub_v6_rate25.csv` — model sinifi atlamasinin etkisi.
2. `sub_v7_rate25.csv` — yeni-ogretmen pseudo dongusunun etkisi.
3. `sub_v67blend_rate25.csv` (varsa +0.2·v3) — cesitlilik etkisi.
4. Kalan haklar: typo-duzeltme rescore'unun v6 uzerine uygulanmasi
   (`rescore_typofix.py` deseni) ve `sub_ens_v234_typofix_rate25.csv`in bekleyen sonucu.
