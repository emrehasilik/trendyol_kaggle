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

## 8) v6/v7 LB sonuclari ve v8: LLM hakem (2026-07-10)

### Sonuclar ve KESIN TANI
- `sub_v6_rate25.csv` = **0.855** (yeni rekor; onceki 0.851)
- `sub_v7_rate25.csv` = 0.852 (yeni-ogretmen pseudo dongusu GERI GITTI)
- Kanit zinciri: model sinifi atlamasi (110M->568M, alaka on-egitimli) sadece
  +0.004; pseudo dongusu -0.003; mimari ensemble daha once +0.000.
  SONUC: tavan MODEL degil ETIKET. Tum modeller ayni etiket soyundan
  (250K gercek pozitif + sentetik/pseudo negatif) ogreniyor; gri bolgede
  (skor 0.05-0.95, ~%7 = ~250K cift) hepsi AYNI hatayi yapiyor. Gercek
  negatif etiket HIC olmadi. 0.855 -> 0.92 icin etiket soyundan BAGIMSIZ
  bilgi kaynagi sart.

### v8 yontemi: yerel LLM hakem (API'siz, A100'de bedava)
- `src/llm_judge_v8.py` — iki asama:
  - `judge --tag v6`: v6 skorlarindan gri bolgeyi secer (cap 400K, karar
    esigine yakin oncelikli), Qwen2.5-32B-Instruct-AWQ (vLLM) ile her cifti
    Evet/Hayir yargilar; P(Evet) ilk-token logprob'lardan cikar. 20K'lik
    chunk'lar Drive'a yazilir -> resumable. `--limit 120` = goz kontrolu
    (dosya yazmaz). Model `TK2_LLM` env ile degisir (hiz: 14B-AWQ).
  - `merge --tag v6 --alpha 0.7`: gri bolgede skor = alpha*LLM + (1-alpha)*CE,
    kesim SABIT %25 pozitif SAYISI (839.920). Cikti:
    `sub_v6_llm{70,100}_rate25.csv`.
- `notebooks/colab_v8_llm.ipynb` — bagimsiz surucu (taze runtime ister;
  vLLM kurulumu torch'u degistirebilir, v6 hucreleriyle ayni oturumda
  KARISTIRMA). Sure A100: kurulum+indirme ~10 dk, yargilama ~1.5-3 saat.
- Submission sirasi: once alpha 0.7 (temkinli), yukselirse alpha 1.0.

### v8 sonrasi yol haritasi
- LB yukselirse v9: LLM etiketleriyle (gri bolge) + guvenli mevcut
  etiketlerle CE'yi yeniden egit (etiket kalitesini kokten duzeltir);
  ayrica LLM hakemi typo-duzeltilmis sorgularla tekrar kosturmak ucuz.
- LB yukselMEZse: LLM kararlarinin ornek incelemesi (limit ciktisi) yol
  gosterir — prompt mu zayif, model mi kucuk, yoksa gri bolge etiketleri
  zaten mi dogru (o zaman tavan veri kalitesinde degil metrik yapisindadir;
  sirada terim-bazli kalibrasyon probe'u var).

## 9) v8 SONUC + tur 2 + v9 plani (2026-07-13)

### v8 LB sonucu: HIPOTEZ DOGRULANDI
- `sub_v6_llm70_rate25.csv` = **0.860**, `sub_v6_llm100_rate25.csv` = **0.861** (rekor).
- Sadece 141K cift (%4.2, gri bolge 0.05-0.95) yargilandi ve +0.006 geldi;
  alpha1.0 > alpha0.7 -> gri bolgede LLM'e TAM guven dogru, CE orada gurultu.
- Etiket-soyu tanisi kesinlesti: bagimsiz bilgi kaynagi (LLM dunya bilgisi)
  model buyutmenin 1.5 kati kazandirdi.

### Sonraki hamleler (kod hazir, notebook hucre 11-15)
- **Tur 2** (`--name v6r2 --lo 0.02 --hi 0.98 --exclude v6`): bant genisletildi,
  yalniz yeni ciftler yargilanir (~30-60 dk); merge `--names v6,v6r2 --suffix r2`
  -> `sub_v6_llm100_rate25_r2.csv`.
- **v9 retrain** (`build_dataset_v9.py --names v6,v6r2`): v5 tabani AYNEN +
  LLM etiketli ciftler x3 tekrar (soft etiket; 0.3<p<0.7 kararsizlar atilir).
  v7-tarzi v6-pseudo KATILMAZ (ayni soy, kanitlanmis sifir katki). Egitim
  `train_ce_v6.py --data train_dataset_v9.parquet --tag v9` (~4-6 saat) ->
  `sub_v9_rate25.csv` (retrain etkisi izole) -> v9 gri bolgesi yargilanir
  (`--tag v9 --name v9 --exclude v6,v6r2`) + merge `--names v6,v6r2,v9`
  -> `sub_v9_llm100_rate25.csv` (ana aday).
- Mantik: yargilanan ciftlerde LLM karari zaten kullaniliyor; v9'un katkisi
  duzeltmeleri YARGILANMAMIS ciftlere genellestirmek. 0.92 yolu bu dongunun
  tekrarindan geciyor (judge -> retrain -> yeni gri bolge -> judge ...).
- Olcekleme secenekleri (gerekirse): 72B-AWQ hakem (A100-80GB sigar, ~2x yavas),
  few-shot prompt, typo-suphelilerin dusuk-skorlu ciftlerini hedefli yargilama.

## 10) v9 SONUC + v10 plani (2026-07-16)

### v9 LB: dongu KANITLANDI
- `sub_v9_rate25.csv` = **0.877** (retrain tek basina +0.010),
  `sub_v9_llm100_rate25.csv` = **0.880** (v9 gri turu +0.003).
- Seri: 0.851 -> v6 0.855 -> LLM r1 0.861 -> r2 0.867 -> v9 0.877 -> 0.880.
- Yorum: LLM etiketleri egitime girince model duzeltmeleri YARGILANMAMIS
  ciftlere genellestiriyor; kazancin buyugu retrain'den geliyor.

### v10 (kod hazir, notebook hucre 16-21)
1. **Tur 3** (`--tag v9 --name v9r2 --lo 0.01 --hi 0.99 --exclude v6,v6r2,v9`):
   bant kiyilari + merge -> `sub_v9_llm100_rate25_r2.csv` (hizli kazanc).
2. **Anchor denetimi** (`llm_audit_train.py`): annvet+v1neg sentetik
   negatifleri (350K) LLM'e sorulur; p>=0.7 false-neg -> soft pozitif
   (`_fix` source), 0.3-0.7 kararsiz atilir. Dosyalar `llm_tr_tr1_*`.
3. **v10 dataseti** (`build_dataset_v10.py --names v6,v6r2,v9,v9r2 --audit tr1`):
   v5 tabani + denetim duzeltmeleri + CELISKI TEMIZLIGI (LLM'in yargiladigi
   ciftlerdeki eski pl_pos/pl_neg atilir) + tum LLM turlari x3 (soft).
4. v10 egitim/skorlama (`--tag v10`) -> `sub_v10_rate25.csv` -> v10 gri
   yargila + tum turlar merge -> `sub_v10_llm100_rate25.csv` (ana aday).
5. **Oran probe**: merge `--rate 0.24 / 0.26` (GPU'suz) — %25 tepesi v3'le
   haritalanmisti, dagilim degisti, yeniden dogrula.

### Teknik notlar
- `llm_judge_v8.py` refactor: `make_llm()` + `pair_texts_from_ids()` ortak;
  audit scripti bunlari kullanir.
- Kota: A100 compute unit bitti -> kullanici Pay-As-You-Go aldi. A100-40GB
  geliyor artik (80 degil); 32B-AWQ hala sigar. L4/T4'e duserse 0. hucrede
  `LLM="Qwen/Qwen2.5-14B-Instruct-AWQ"`.
- vLLM kurulum tuzaklari cozuldu (hucre 6): uv --torch-backend=auto ile
  zorla reinstall + libnvrtc.so.13 sistem yoluna symlink; gercek test
  `from vllm import LLM` (duz `import vllm` tembel, yaniltir).
