"""Tum path'ler, hiperparametreler ve seed'ler tek yerde."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# Colab tasinabilirligi: TK2_WORK verilirse artifacts/models/output oraya gider
# (Drive kaliciligi icin), TK2_DATA veri klasorunu ezer. Lokal Windows'ta env
# degiskenleri yoksa davranis eskisiyle birebir ayni.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ["TK2_DATA"]) if os.environ.get("TK2_DATA") \
    else PROJECT_DIR / "data"
if os.environ.get("TK2_WORK"):
    _WORK = Path(os.environ["TK2_WORK"])
    ARTIFACTS_DIR = _WORK / "artifacts"
    OUTPUT_DIR = _WORK / "output"
    MODELS_DIR = _WORK / "models"
else:
    ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
    OUTPUT_DIR = PROJECT_DIR / "output"
    MODELS_DIR = PROJECT_DIR / "models"

# Eski projenin hazir varliklari (1.5 GB embedding'ler kopyalanmaz, path ile okunur)
OLD_PROJECT_DIR = Path(r"C:\Users\Hp\Desktop\project\trendyol_kaggle")
OLD_MODELS_DIR = OLD_PROJECT_DIR / "models"
FEATURE_BUILDER_PKL = OLD_MODELS_DIR / "feature_builder.pkl"
# Satir sirasi terms.csv / items.csv CSV sirasiyla birebir ayni
# (eski train.py: fb.fit(load_terms(...), load_items(...)))
TERM_EMB_NPY = OLD_MODELS_DIR / "feature_builder.pkl.term_emb.npy"
ITEM_EMB_NPY = OLD_MODELS_DIR / "feature_builder.pkl.item_emb.npy"

ITEMS_CSV = DATA_DIR / "items.csv"
TERMS_CSV = DATA_DIR / "terms.csv"
TRAINING_PAIRS_CSV = DATA_DIR / "training_pairs.csv"
SUBMISSION_PAIRS_CSV = DATA_DIR / "submission_pairs.csv"

SEED = 42

# ---------------------------------------------------------------- Faz 1: mining
ANN_TOP_K = 200                 # term basina saklanan en yakin item sayisi
ANN_SKIP_TOP = 5                # en yakin 5 sira hard-negatif havuzundan atlanir (false-neg korumasi)
POS_SIM_PERCENTILE = 25         # embed_sim >= termin pozitif simlerinin bu persentili ise negatif sayilmaz
HARD_PER_POS = 2                # k pozitif icin 2k hard
MEDIUM_PER_POS = 1              # k pozitif icin 1k medium (ayni yaprak kategori)
EASY_PER_POS = 1                # k pozitif icin 1k easy (rastgele)
MAX_LIST_PER_TERM = 150         # cok pozitifli termlerde liste tavani

# ---------------------------------------------------------------- Faz 1c: LB-proxy validasyon
N_PROXY_TERMS = 2000            # her seyden ayrilan term sayisi
PROXY_LIST_SIZE = 100           # test yapisini taklit eden aday listesi boyu

# ---------------------------------------------------------------- Faz 3: cross-encoder
CE_MODEL_NAME = "dbmdz/bert-base-turkish-128k-uncased"
CE_MAX_LENGTH = 128
CE_N_FOLDS = 3
CE_EPOCHS = 2
CE_LR = 2e-5
CE_WEIGHT_DECAY = 0.01
CE_WARMUP_RATIO = 0.06
CE_BATCH_SIZE = 32
CE_GRAD_ACCUM = 2
CE_INFER_BATCH_SIZE = 256
# Colab'da HF cache Drive kotasini yememesi icin lokal diske alinabilir
HF_CACHE = os.environ.get("TK2_HF_CACHE", str(MODELS_DIR / "hf_cache"))

# ---------------------------------------------------------------- v6: Colab buyuk govde
# Karar gerekcesi HANDOFF.md bolum 7'de. TK2_MODEL ile govde degistirilebilir
# (yedek aday: microsoft/mdeberta-v3-base — yalniz bf16 destekli GPU'da).
CE_V6_MODEL_NAME = os.environ.get("TK2_MODEL", "BAAI/bge-reranker-v2-m3")
CE_V6_MAX_LENGTH = 192          # cift (sorgu+item+ozel tokenlar) ust siniri
CE_V6_MAX_ITEM_TOKENS = 160     # zengin item metni icin genis pay
CE_V6_MAX_TERM_TOKENS = 32
CE_V6_EPOCHS = 2
CE_V6_LR = 1e-5                 # buyuk govde icin dusuk lr (base'de 2e-5 idi)
CE_V6_WEIGHT_DECAY = 0.01
CE_V6_WARMUP_RATIO = 0.06
CE_V6_EFF_BATCH = 64            # efektif batch sabit; bs/accum GPU'ya gore bolunur
CE_V6_INFER_BATCH = 384
CE_V6_HOLDOUT_FRAC = 0.02       # terim bazli izleme seti (model SECIMI yapilmaz)
CE_V6_CKPT_MINUTES = float(os.environ.get("TK2_CKPT_MIN", "20"))

for _d in (ARTIFACTS_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
